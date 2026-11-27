# Inductor 慢 Triton kernel 定位与修复分析（27B 三个、VL 一个）

日期：2026-11-27。环境：A5-25，Ascend950DT（64 个向量核），torch 2.13、vLLM/vllm-ascend 0.33，容器 `inductor_214build_z00896543_20261120_1546`。
模型：Qwen3.8-27B-w8a8、Qwen2.5-VL-7B-Instruct，TP1，1024 捕获。

本文记录 vllm-ascend Inductor 路径上四个生成 Triton kernel 的定位、修复和验证结果。对比对象是 default 编译路径（vLLM Ascend 原有的 AscendCompiler 路径）。修复落在 torch_npu 的 `triton_experimental` 后端，包括静态启动器修复、clamp 证明器补充和纯搬运 fp32 优化。

## 1. 问题

32K prefill 的 Profile 窗口里，Inductor 路径有四个生成 kernel 明显过慢。下表是定位前从模型 Profile 读到的累计耗时，default 列是同一窗口里完成同一计算的原生算子：

| 序号 | kernel | 模型 | Inductor | default |
|---|---|---|---|---|
| 1 | `triton_unk_fused_mul_npu_dynamic_quant_sigmoid_view_0` | 27B | 1201 ms | Sigmoid + Mul，92 ms |
| 2 | `triton_unk_fused_zeros_4` | 27B | 590 ms | ZerosLike，41 ms |
| 3 | `triton_unk_fused__unsafe_view_clone_view_0` | 27B | 1374 ms | 待确认 |
| 4 | `triton_unk_fused_copy_split_with_sizes_3` | VL | 2788 ms | 待确认 |

局部算子的目标是：同一形状下生成的 kernel 不慢于 default 中完成同一计算的算子之和，decode 短形状不变慢。模型级目标是：32K 输入的吞吐和 TTFT 加速比（Inductor 对 default）比修复前提高，8K 不下降，其余负载的中位数加速比下降不超过 2%。

## 2. 结论

- Zeros、Sigmoid+Mul、GDN gate 重排和 QKV 回写都受到静态启动器大 grid 问题影响。采用社区 `!50025` 后，物理 block 数恢复为最多 64 个，逻辑 grid 保持完整，不需要改变 A5 kernel 的生成方式。
- QKV 的 K 切片读取还受到冗余 clamp 影响。证明器补充识别 `npu_triton_helpers.ge/gt(...)` 后，地址恢复仿射，单独该项修复使 QKV 从 `803.8 us` 降到 `230.5 us`。
- QKV 的纯搬运融合组存在 bf16 → fp32 → bf16 的无意义往返。覆盖上游 load 级 cast 后，在 `!50025 + clamp` 基线上，2048 token 的 QKV 进一步从 `33.826 us` 降到 `18.552 us`，其他形状也有稳定收益。
- 四个 kernel 的数值检查均通过，长形状均达到原定局部验收目标。两个模型固定输入的 eager/Inductor 输出均与各自 prefix 逐字节相同，说明这些修复没有引入额外的输出变化。
- 两个模型的两轮完整性能验证仍在运行，本文暂不给出最终模型级性能结论。QKV 全 4608 通道读写保留为后续优化空间。

## 3. 方法

- **对照目标**：按 kernel 形状和调用链对齐 default 路径的 32K prefill Profile（`locate/default_targets/REPORT.md`）。

  | kernel | 形状 | default 里的算子 | 验收目标（每次调用） |
  |---|---|---|---:|
  | Sigmoid+Mul | 8192 token | Sigmoid 78.11 + Mul 101.04 | 不超过 179.2 us |
  | Zeros | 8192 token | ZerosLike | 不超过 26.0 us |
  | GDN gate 重排 | 8192 token | aclnnInplaceCopy_Slice | 不超过 88.7 us |
  | QKV 读改写 | 2048 token | 10 个 aclnnInplaceCopy，合计 69.86 us | 不超过 69.86 us |

- **最小复现**：每个 kernel 使用只含目标计算的 PyTorch 函数，通过 `torch.compile(backend="inductor")` 编译，并与模型 Profile 中的生成代码比对。四个目标 kernel 体一致，差异仅限与进程环境有关的元数据。
- **计时**：使用 `torch_npu.profiler`（Level 1、PipeUtilization）导出的 `kernel_details.csv`，读取目标 kernel 的 `Duration(us)` 中位数。原生对照在同一进程中用相同方法测量。
- **环境隔离**：使用独立安装目录加载修复后的 torch_npu，不依赖主 conda 中可能残留的实验修改。最终模型测试在精度和性能启动阶段都检查实际加载路径。
- **规模**：每个 kernel 测五个 token 规模，QKV 测四个，包含模型长形状和 decode（8 token）。模型性能先跑两轮，结果稳定时不追加第三轮。

## 4. 定位结论

### 4.1 静态启动器没有遵守 AutoBlockify 的启动约定

A5 的 AutoBlockify 形态需要区分逻辑 grid 和物理 block 数：

- 逻辑 grid 表示完整的逻辑任务空间，必须传给 kernel；
- 物理 block 数应限制在实际可承载的范围内，一个物理 block 可以循环处理多个逻辑任务。

triton-ascend 普通启动器会配合 AutoBlockify 限制物理 block 数；torch_npu 静态启动器消费同一份编译后的 kernel，却直接按逻辑 grid 启动大量物理 block。于是 Zeros、Sigmoid+Mul、GDN 分别出现 1536、2305、1639 个 block，每个 block 只处理一个 tile，固定调度开销被放大。

同一 kernel 体仅更换启动器就能恢复到 64 个 block 和接近原生算子的耗时，说明主要问题在启动方式，而不是 tile、下标形式或数学表达式。

### 4.2 clamp 证明器不识别调用形式的掩码

QKV 给 `new_k` 的读取地址套了防御性的 `tl.maximum(..., 0)`，使地址不再是仿射式，triton-ascend 将其编译成 `triton_indirect_load`，退化为逐元素读取。

这个 clamp 本可由 `nonnegative_clamp_mode="provable"` 证明跳过：有效 lane 上 `3584 <= x0 < 4096`，因此 `-3584 + x0 + 512*x1 >= 0`。但原证明器只识别 Python `>=` 比较，后端实际生成的是 `npu_triton_helpers.ge(...)`，导致证明失败并保留 clamp。

单独应用 clamp 修复时，QKV 从 `803.8 us` 降到 `230.5 us`。剩余启动器问题由 `!50025` 处理。

### 4.3 纯搬运 fp32 提升走了另一条生成路径

纯搬运融合组不做浮点算术，bf16 → fp32 → bf16 是无损往返，只会增加转换指令和片上资源压力。

原实现只在 `npu_triton_compute_type()` 中跳过这类提升，但上游 `TritonKernel.load()` 会直接给 load 字符串追加 `.to(tl.float32)`，只读取 `config.triton.codegen_upcast_to_fp32`，不调用 `npu_triton_compute_type()`。因此，即使 QKV 融合组已被判定为纯搬运，三处 load 仍保留 fp32 转换。

社区 `!48031` 也明确区分了上游 load 级 cast 和 NPU compute-type 重映射，但没有提供按 kernel 跳过纯搬运提升的实现。

### 4.4 QKV 全通道读写来自完整的 slice_scatter 语义

Inductor 将 Q/K 的局部 `copy_` 函数化为完整 `slice_scatter` 并融合，形成“读取完整旧 QKV、更新 Q/K、写回完整 QKV”的形式。V 虽未被修改，也被读取并原样写回。

default 没有执行这项融合优化，而是保留原生切片 `copy_`，只读取源数据并覆盖 Q/K 目标视图，不访问 V。全通道问题因此属于图级优化缺口，不是索引或数值实现错误。修复必须同时缩小读取、写回和迭代范围，不能只删除 V 的 load。

## 5. 修复

### 5.1 采用社区静态启动器修复 !50025

社区提交 `ad2224d30a23c0209dbcc096434065a19568575f` 已合入 master，本地 cherry-pick 为 `8207bd55be`，A5 主源码目录为 `c31aa61dae`。

该修复只限制 CANN 实际启动的物理 block 数，同时保留传给 kernel 的完整逻辑 grid，不修改 A5 kernel 的生成方式，也不引入新的用户开关。静态启动器单元测试 `84 passed`，fast launcher `19 passed`。

### 5.2 clamp 证明器识别 helper-call 掩码

社区 `!46173` 已合入通用的可证明非负 clamp 跳过机制，本次补充的是它尚未覆盖的 helper-call 生成形式；当前社区 master 尚未包含这项补充。

`codegen/provable_clamp.py` 做了两项补充：

1. `_npu_mask_conj_atoms` 识别 `npu_triton_helpers.ge/gt(a, b)`：左侧必须是轴变量，右侧必须是整数常量；`ge` 提供下界 `k`，整数 `gt` 提供下界 `k+1`。
2. `_npu_ast_eval` 增加整数专用求值路径，只允许整数到整数转换和带整数 dtype 的常量。

上界、反向操作数、其他模块的同名函数、浮点、bool/int1、关键字参数和未知表达式均不产生下界；证明失败或异常时保留 clamp。没有新增配置，仍可通过 `nonnegative_clamp_mode="on"` 恢复旧行为。

新增测试套件内部 48 条，相关 pytest 入口 `2 passed`。同一套件在修改前有 22 个应当证明成功的用例失败。

### 5.3 纯搬运 kernel 跳过 load 级 fp32 提升

在现有 `NPUTritonKernel.load()` 包装中，仅对 `_npu_pure_data_movement=True` 的 kernel 使用 `config.patch("triton.codegen_upcast_to_fp32", False)` 包住上游 load 调用，返回后恢复配置。算术、归约和真实 dtype 转换路径保持原行为。

判定还检查浮点输入 dtype 与输出是否一致，避免把 bf16 → fp16 等真实转换误判为纯搬运。最终单元测试 `9 passed`，覆盖 load 级开关、配置恢复、非纯搬运不跳过和混合输入 dtype 拒绝等情况。

## 6. kernel 级结果

以下结果来自同一张卡上的最小复现，均为 msprof 每次调用的设备时间。原始列用于说明问题量级，当前列是 `!50025 + clamp + 纯搬运修复` 的结果；GDN 未命中纯搬运优化，Zeros 和 Sigmoid+Mul 也不受其影响。

| kernel | token 数 | 原始 | 当前 | 原生对照 |
|---|---:|---:|---:|---:|
| Zeros bf16 [t,48,128] | 8192 | 378.4 us | 24.0855 us | 22.076 us |
| | 2048 | 28.7 us | 8.0575 us | 7.972 us |
| | 512 | 4.5 us | 3.772 us | 4.307 us |
| | 128 | 2.3 us | 2.227 us | 3.443 us |
| | 8 | 1.4 us | 1.544 us | 2.1245 us |
| Sigmoid+Mul+动态量化 [t,6144] | 8192 | 2338.9 us | 115.6465 us | 157.617 us |
| | 2048 | 137.7 us | 25.4015 us | 24.943 us |
| | 512 | 15.1 us | 8.6245 us | 11.1685 us |
| | 128 | 4.3 us | 4.2265 us | 7.180 us |
| | 8 | 3.6 us | 3.6295 us | 4.0645 us |
| GDN gate 重排（带步长的 clone） | 8192 | 868.8 us | 66.1575 us | 66.707 us |
| | 2048 | 52.1 us | 13.4185 us | 9.2705 us |
| | 512 | 6.2 us | 4.7525 us | 5.194 us |
| | 128 | 2.9 us | 2.845 us | 3.532 us |
| | 8 | 1.8 us | 1.7945 us | 3.517 us |
| QKV 回写 [t,4608] | 2048 | 803.8 us | 18.552 us | 10.3695 us（两次切片拷贝） |
| | 512 | 61.7 us | 6.1925 us | 7.084 us（两次切片拷贝） |
| | 128 | 6.9 us | 3.094 us | 6.123 us（两次切片拷贝） |
| | 8 | 5.6 us | 2.104 us | 4.6495 us（两次切片拷贝） |

注：QKV 的局部验收目标是 default 模型路径中 10 个相关 kernel 合计的 `69.86 us`，不是单独两次切片拷贝的耗时。当前 `18.552 us` 同时低于这两个对照。

四个 kernel 的数值检查全部通过：Zeros 在输出块预填 1 后检查全零，GDN 与 eager 逐值相等，QKV 各形状数值差为 0，Sigmoid+Mul 与既有 eager 检查一致。

长形状验收结果：

| kernel | 当前 | 验收目标 | 结果 |
|---|---:|---:|---|
| Zeros | 24.0855 us | 不超过 26.0 us | 达到 |
| Sigmoid+Mul | 115.6465 us | 不超过 179.2 us | 达到 |
| GDN gate 重排 | 66.1575 us | 不超过 88.7 us | 达到 |
| QKV 回写 | 18.552 us | 不超过 69.86 us | 达到 |

纯搬运修复的独立增量如下。基线是 `!50025 + clamp + 未生效的纯搬运实现`，因此可以直接体现 load 级修复及重新 autotune 后的整体效果：

| token 数 | 基线 | 当前 | 时延降低 |
|---:|---:|---:|---:|
| 2048 | 33.826 us | 18.552 us | 45.2% |
| 512 | 10.8465 us | 6.1925 us | 42.9% |
| 128 | 4.832 us | 3.094 us | 36.0% |
| 8 | 4.1015 us | 2.104 us | 48.7% |

2048 token 的 autotune XBLOCK 从 16384 调整为 12128，所以这组数据不是固定 tile 下只删除 cast 的微基准。

## 7. 模型级验证

最终验证使用 `install_pure`，精度和性能进程均确认加载该隔离目录，而不是主 conda。

- **VL**：固定输入的 eager/Inductor 输出与 `vl_prefix` 逐字节相同，性能冒烟通过；两轮完整性能正在运行。
- **27B**：固定输入的 eager/Inductor 输出与 `qwen27b_prefix` 逐字节相同，性能冒烟通过；两轮完整性能正在运行。

两个模型的结果目录分别为 `results/vl_vl_pure_final` 和 `results/qwen27b_qwen27b_pure_final`。待两轮结束后，使用 `compare_perf.py --repeats 2` 与各自 prefix 比较，再补充最终模型级结论。

## 8. 交付物

| 内容 | 位置 |
|---|---|
| 静态启动器、clamp、纯搬运补丁 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/patches/` |
| 纯搬运根因与结果摘要 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/PURE_DATA_MOVEMENT_RESULT.md` |
| 定位与实验脚本 | `server-scripts/benchmark-diagnosis/20261126_triton_kernel_perf/` |
| kernel 原始结果 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/server/locate/k2_zeros/results_sizes/` |
| 模型精度、性能结果和日志 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/server/results/`、`server-results/benchmark-diagnosis/20261126_triton_kernel_perf/logs/` |
| 工作过程记录 | `server-scripts/benchmark-diagnosis/20261126_triton_kernel_perf/WORKLOG.md` |

测试汇总：静态启动器 `84 passed`，fast launcher `19 passed`，clamp 相关 pytest 入口 `2 passed`，纯搬运 `9 passed`。四个目标 kernel 的数值检查均通过。

## 9. 未解决的问题

1. **两个模型的最终两轮性能尚未完成。** 当前只能确认固定输入输出未变和性能冒烟通过，不能提前宣称模型级性能验收完成。
2. **GDN 在中等形状上仍慢于原生拷贝。** 2048 token 当前为 `13.4185 us`，原生为 `9.2705 us`。GDN 也未命中当前纯搬运判定，生成代码仍保留 fp32 转换；是否需要调整融合组判定或 tile 候选，留待后续分析。
3. **QKV 仍读写全部 4608 通道。** 2048 token 时当前总流量约 `54.52 MB`，只复制 Q/K 的形式约 `33.55 MB`。该优化涉及 mutation、别名、迭代域和调度，按当前任务决定暂缓。
4. **其他覆盖范围未扩展。** `cpp_wrapper` 和 AscendForge 门禁未在本次验证中运行，不将其报告为通过。

## 修订记录

- 2026-11-27：将静态启动器方案更新为社区 `!50025`；补充 clamp 的社区背景与验证；加入纯搬运 load 级根因、修复和四形状收益；更新四个 kernel 的最终结果、模型精度状态及后续未解决项。

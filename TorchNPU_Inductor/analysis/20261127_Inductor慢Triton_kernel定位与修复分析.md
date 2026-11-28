# Inductor 慢 Triton kernel 定位与修复分析（Qwen3.8-27B-w8a8 三个、Qwen2.5-VL-7B-Instruct 一个）

日期：2026-11-27。环境：A5-25，Ascend950DT（64 个向量核），torch 2.13、vLLM/vllm-ascend 0.33，容器 `inductor_214build_z00896543_20261120_1546`。
模型：Qwen3.8-27B-w8a8、Qwen2.5-VL-7B-Instruct，TP1，1024 捕获。

本文记录 vllm-ascend Inductor 路径上四个生成 Triton kernel 的定位、修复和验证结果。对比对象是 default 编译路径（vLLM Ascend 原有的 AscendCompiler 路径）。修复落在 torch_npu 的 `triton_experimental` 后端，包括静态启动器修复、clamp 证明器补充和纯搬运 fp32 优化。

## 1. 问题

32K prefill 的 Profile 窗口里，Inductor 路径有四个生成 kernel 明显过慢。下表是定位前从模型 Profile 读到的累计耗时，default 列是同一窗口里完成同一计算的原生算子：

| 序号 | kernel | 模型 | Inductor | default |
|---|---|---|---|---|
| 1 | `triton_unk_fused_mul_npu_dynamic_quant_sigmoid_view_0` | Qwen3.8-27B-w8a8 | 1201 ms | Sigmoid + Mul，92 ms |
| 2 | `triton_unk_fused_zeros_4` | Qwen3.8-27B-w8a8 | 590 ms | ZerosLike，41 ms |
| 3 | `triton_unk_fused__unsafe_view_clone_view_0` | Qwen3.8-27B-w8a8 | 1374 ms | 待确认 |
| 4 | `triton_unk_fused_copy_split_with_sizes_3` | Qwen2.5-VL-7B-Instruct | 2788 ms | 待确认 |

局部算子的目标是：同一形状下生成的 kernel 不慢于 default 中完成同一计算的算子之和，decode 短形状不变慢。模型级目标是：32K 输入的吞吐和 TTFT 加速比（Inductor 对 default）比修复前提高，8K 不下降，其余负载的中位数加速比下降不超过 2%。

## 2. 结论

- Zeros、Sigmoid+Mul、GDN gate 重排和 QKV 回写都受到静态启动器大 grid 问题影响。采用社区 `!50025` 后，物理 block 数恢复为最多 64 个，逻辑 grid 保持完整，不需要改变 A5 kernel 的生成方式。
- QKV 的修复链为：原始 `803.8 us` → 只修 clamp `230.5 us` → `!50025 + clamp` `33.826 us` → 再加纯搬运修复 `18.552 us`。clamp 消除 K 的逐元素读取，`!50025` 限制物理 block 数，纯搬运修复再去掉 bf16 → fp32 → bf16 的无意义往返；其他形状也有稳定收益。
- 四个 kernel 的数值检查均通过，长形状均达到原定局部验收目标。两个模型固定输入的 eager/Inductor 输出均与各自 prefix 逐字节相同，说明这些修复没有引入额外的输出变化。
- Qwen2.5-VL-7B-Instruct 两轮完整性能满足全部验收条件。Qwen3.8-27B-w8a8 的 32K 吞吐、TPOT、TTFT 均比 prefix 改善，但 ShareGPT QPS1 的 TTFT 两轮汇总值从 `1.024` 降至 `0.986`，回落超过 2%，因此不能判定其全部通过。Qwen2.5-VL-7B-Instruct 的最终 32K TTFT 两轮分别为 `0.983`、`0.998`，Qwen3.8-27B-w8a8 分别为 `0.976`、`0.990`，仍是略慢于 default，而不是已经全面追平。
- QKV 全 4608 通道读写保留为后续优化空间。

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

#### 从 PyTorch 到 NPU 执行要经过哪些环节

这几个慢 kernel 的完整链路是：

```text
PyTorch 模型
  → Inductor 生成 Triton Python kernel
  → triton-ascend / AscendNPU-IR 编译成 NPU 二进制
  → C++ launcher 准备参数并调用 CANN
  → CANN 把任务提交到 NPU
  → NPU 执行 kernel
```

其中：

- **Triton kernel** 是在 NPU 上执行的设备程序，真正完成 load、算术和 store。
- **tile** 是 kernel 一次处理的一小块数据。大张量会被划分成很多 tile。
- **program / block** 是执行这些 tile 的任务实例。本文用“逻辑 grid”表示完整任务空间，用“物理 block 数”表示实际提交给硬件执行的任务数。
- **CANN** 是底层运行时和算子软件栈，launcher 通过它把 kernel 和参数提交到设备。
- **launcher（启动器）** 是 CPU 侧的调用代码，不执行 Triton 中的数学表达式。它负责加载二进制、准备参数、决定启动数量并调用 CANN。

普通 triton-ascend C++ launcher 和 torch_npu 静态 C++ launcher 是同一位置上的两种可替代实现，不是父子关系，也不是先后串联调用。它们消费同一份 kernel 二进制，但必须遵守同一份启动约定。静态启动器的价值是减少 Python/JIT 路径的主机侧开销，所以修复目标是保留它，而不是绕回普通启动器。

#### AutoBlockify 是怎样进入这条链路的

AutoBlockify 不是本次修复新引入的功能，也不是静态启动器自己的特性。从代码演变关系看，这条链路是逐步形成的：

1. **A5 先采用 one-program-per-tile 的生成形态。** Inductor 生成的 Triton 代码用 `group_size = 1`、`group_base = group_id` 表示一个逻辑 program 对应一个 tile，逻辑 grid 因而等于 tile 总数。这种写法直接、便于按逻辑任务调度，但如果原样启动全部 program，大形状下会产生很多 block。
2. **编译阶段通过 AutoBlockify 支持把多个逻辑任务映射到有限的物理 block。** AutoBlockify 属于 triton-ascend/AscendNPU-IR 编译链。经过它处理后，物理 block 可以按照 kernel 收到的完整逻辑 grid 循环覆盖工作项，不再要求“一个逻辑 tile 必须对应一个物理 block”。
3. **普通 triton-ascend launcher 在调用期配合这个编译结果。** 它根据 `auto_blockify_enabled` 等元数据限制实际物理 block 数，同时把完整逻辑 grid 传给 kernel。普通 launcher 只是遵守编译器给出的启动约定，并不拥有 AutoBlockify 这一编译功能。
4. **后来加入 torch_npu 静态 launcher。** 静态 launcher 为减少主机侧调用开销，复用了同一份 kernel 二进制和参数 ABI，却没有完整复现普通 launcher 对 AutoBlockify 的处理：它提取了 kernel、参数和逻辑 grid，但把逻辑 grid 同时当成 CANN 的物理 `blockCount`。

所以，问题的因果链不是“AutoBlockify 生成了奇怪 kernel”，而是：

```text
A5 的逻辑任务生成方式
  + AutoBlockify 提供的逻辑/物理任务映射能力
  + 普通 launcher 原本正确的物理 block 限制
  - 静态 launcher 复用时漏掉这项限制
  = 同一份 kernel 在静态 launcher 下被启动了过多物理 block
```

静态 C++ 中其实分别存在两个信息：一个是传给 CANN 的物理 `blockCount`，另一个是打包给 kernel 的逻辑 grid。`!50025` 只限制前者，后者保持不变，因此不需要改变 A5 kernel 的生成方式。

#### AutoBlockify 为什么要求区分逻辑 grid 和物理 block

它把逻辑任务空间和实际执行资源解耦：一个物理 block 可以在编译后的 kernel 内循环处理多个逻辑 tile。

以 Zeros 的 1536 个逻辑 tile 为例，正确的执行方式是：

```text
逻辑 grid：1536
  → 告诉 kernel 完整任务空间，保证每个 tile 都能被覆盖

物理 block：最多 64
  → 对应当前 A5 的 64 个向量核
  → 每个物理 block 平均处理约 1536 / 64 = 24 个 tile
```

这里两个数量缺一不可：

- 如果只把 grid 截断为 64，却不给 kernel 完整逻辑 grid，编号 64 之后的 tile 就不会执行，可能直接造成输出缺失。
- 如果直接启动 1536 个物理 block，虽然能覆盖所有 tile，但大量 block 只处理一个 tile，任务创建、排队和调度开销会反复出现。

因此，正确做法不是简单“把 grid 改小”，而是**限制物理启动数，同时保留完整逻辑 grid**。

#### 静态启动器具体错在哪里

A5 的这种 kernel 形态先于静态启动器出现，原本依赖 AutoBlockify 和普通 launcher 配合。普通 launcher 会根据 AutoBlockify 的元数据限制物理 block 数；静态启动器后来复用了同一份二进制和逻辑 grid，却没有补上这项限制，直接把逻辑 grid 当成物理启动数。

最终形成：

```text
编译器预期：64 个物理 block 循环覆盖完整逻辑 grid
静态启动器实际：1536 / 2305 / 1639 个物理 block 各自只做一个 tile
```

Zeros、Sigmoid+Mul、GDN 的模型 Profile 分别显示 1536、2305、1639 个 block。同一 kernel 体只换用普通 triton-ascend launcher，就能恢复到 64 个 block 和接近原生算子的耗时，说明瓶颈在启动方式，不在 tile 形状、sigmoid 数学形式或 clone 下标写法。

这也解释了为什么问题看起来只出现在部分 kernel：

- 逻辑 grid 本来不超过 64 时，多波次调度问题不明显；
- 使用普通 launcher 时，物理 block 数本来就会被限制；
- 不经过 A5 + AutoBlockify + 静态启动器这条组合路径时，不会触发同一缺口；
- 满足这条组合路径且逻辑 grid 很大的 kernel，则都可能暴露同类问题。

`!50025` 正是在静态 launcher 内补齐普通 launcher 已遵守的约定，不需要修改 Triton kernel，也不需要增加 A5 专用循环或用户开关。它只改变 CANN 的物理启动数，完整逻辑 grid 仍传给 kernel，因此不会影响其他生成路径。

#### 为什么没有采用其他方案

- **只截断 grid**：会漏掉后续逻辑 tile，不能作为通用修复。
- **改写 A5 kernel 为均分循环**：也能恢复性能，但改变了共享 kernel 的生成方式，影响面大于病灶，而且需要额外处理普通 launcher、cpp_wrapper 和 rsplit 等形态。
- **旁路到普通 launcher**：放弃静态启动器的主机侧收益，decode 小形状还可能增加启动开销。

社区 `!50025` 选择在真正缺失约定的静态 launcher 中修复，避免了这些额外影响。

### 4.2 clamp 证明器不识别调用形式的掩码

#### 为什么会有这个 clamp

QKV 的 4608 个通道可以分成：

```text
Q：[0, 3584)
K：[3584, 4096)
V：[4096, 4608)
```

K 在原始 QKV 中从 3584 开始，但 `new_k` 自身只有 512 个通道。生成 kernel 在完整 QKV 的 4608 通道迭代域中读取 `new_k` 时，要把 QKV 的坐标换算成 K 局部坐标：

```python
index = -3584 + x0 + 512*x1
```

其中 `x0` 是通道轴，`x1` 是 token 轴。完整迭代域中还包含 Q 和 V 的 lane，所以只有 K 范围内的 lane 应该读取 `new_k`：

```python
mask = (x0 >= 3584) & (x0 < 4096) & 其他有效形状掩码
value = tl.load(new_k + index, mask=mask)
```

**lane** 可以理解为向量任务中的一个元素位置，**mask** 是每个 lane 上的布尔条件。有效 lane 允许读取，无效 lane 不应实际访存。4608 个通道里只有 512 个属于 K，约 88.9% 的 lane 会被这个范围掩码排除。

对不在 K 范围内的 lane，`-3584 + x0` 可能为负。历史上的 triton-ascend 对 masked lane 的负地址存在编译限制或异常风险，因此后端曾添加防御性代码：

```python
index = tl.maximum(index, 0)
```

它的目的只是保护无效 lane，有效 K lane 的地址本应保持不变。因此不能不加区分地全局删除 clamp；必须证明有效 lane 上地址已经非负。

#### 为什么冗余 clamp 会把块读取变成逐元素读取

**仿射地址**是由常数和轴变量通过加减乘形成的规则地址，例如：

```text
base + x0 + 512*x1
```

编译器可以根据这种形式识别连续段、固定步长和访问范围，从而生成块拷贝。`tl.maximum(index, 0)` 引入分段选择后，地址不再是单纯的仿射表达式，编译器难以继续做同一套证明，于是退化为 `triton_indirect_load`，按 lane 分别计算地址并逐元素读取。

这就形成了一种很遗憾的情况：clamp 在有效 lane 上数学上完全不起作用，却在编译阶段改变了整个 load 的实现方式，而且完整 `[每块行数, 4608]` 区域都会受到影响。

#### 为什么可以证明它是冗余的

有效 K lane 满足：

```text
3584 <= x0 < 4096
x1 >= 0
```

所以：

```text
-3584 + x0 >= 0
512*x1 >= 0
因此 index = -3584 + x0 + 512*x1 >= 0
```

在这些 lane 上：

```python
tl.maximum(index, 0) == index
```

删除 clamp 不会改变任何有效读取，只是让编译器重新看到仿射地址。无效 lane 则依赖 masked load 不实际访存，这也是社区原 provable 模式已经采用的前提。

#### 原证明器为什么没能完成这项证明

证明器读取的是生成 DSL 的 **AST（抽象语法树）**，也就是代码的语法结构，而不是直接读取最初的 PyTorch FX 图。相同语义可以有不同的 AST：

```python
x0 >= 3584                              # ast.Compare，Python 比较节点
npu_triton_helpers.ge(x0, 3584)          # ast.Call，函数调用节点
```

原实现只处理 `ast.Compare`，后端却会把 IR 比较统一改写成 `npu_triton_helpers.ge/gt/lt/le(...)`。因此它虽然拥有完整掩码，却无法从中提取 `x0 >= 3584` 这个下界，最终记录 `provable_unproven=1` 并保留 clamp。

证明过程可以简化为：

```text
从 mask 的合取条件中找轴下界
  → 得到 x0 >= 3584
  → 将下界代入 index 的线性表达式
  → 证明有效 lane 上 index >= 0
  → 返回“可以跳过 clamp”
```

本次修复只是让第一步也能识别 helper-call，后面的线性证明没有改变。`ge` 提供下界 `k`，整数 `gt` 提供下界 `k+1`；浮点、bool/int1、反向操作数、未知函数、关键字参数和无法确认的转换都拒绝证明。证明失败或异常时继续保留 clamp，所以主要失败模式是漏优化，而不是放宽正确性条件。

#### 与其他修复的耗时关系

QKV 同时存在多个独立瓶颈，不能只截取两个数字直接比较。2048 token 下各组合的关系是：

| 修复组合 | 耗时 | 主要变化 |
|---|---:|---|
| 原始 | 803.8 us | clamp 导致间接读取，静态启动器大 grid，load 提升到 fp32 |
| 只修 clamp | 230.5 us | K 恢复块读取，启动器问题仍在 |
| 只修 `!50025` | 70.60 us | 物理 block 限制到 64，clamp 和 fp32 提升仍在 |
| `!50025 + clamp` | 33.826 us | 块读取和启动方式都恢复，重新 autotune；fp32 提升仍在 |
| 再加纯搬运修复 | 18.552 us | load 不再提升到 fp32，重新 autotune |

这些数字来自不同补丁组合，不能按倍数简单相加。它们用于说明每项修复解决了哪一层浪费，以及组合后为什么能达到最终耗时。

### 4.3 纯搬运 fp32 提升走了另一条生成路径

#### 先看这个 kernel 到底读写了哪些内存

每个 token 都有一行 4608 个 bf16 元素。kernel 使用的三个输入输出指针分别对应：

| 生成变量 | 指针 | 实际内存 | 读取范围 |
|---|---|---|---|
| `tmp9` | `in_out_ptr0` | 原始 QKV，形状 `[token, 4608]` | 整行 4608 个通道 |
| `tmp8` | `in_ptr1` | `new_q`，形状 `[token, 3584]` | 只在 Q 范围内读取 |
| `tmp6` | `in_ptr0` | `new_k`，形状 `[token, 512]` | 只在 K 范围内读取 |

三次读取生成的是三组临时值：

```python
# 读取原始 QKV。后面 V 区域要保留旧值，所以需要这个 old。
tmp9 = tl.load(in_out_ptr0 + (x0 + 4608*x1), mask)

# 读取 new_q。x1 是 token 编号，x0 是 Q 内的通道编号。
tmp8 = tl.load(in_ptr1 + (x0 + 3584*x1), q_mask, other=0.0)

# 读取 new_k。QKV 中的 K 从 3584 开始，所以地址要减去 3584。
tmp6 = tl.load(in_ptr0 + (-3584 + x0 + 512*x1), k_mask, other=0.0)
```

随后没有乘加、sigmoid 或其他浮点算术，只有两次选择：

```python
# Q 范围使用 new_q，其他范围暂时保留原始 QKV。
tmp10 = tl.where(x0 < 3584, tmp8, tmp9)

# K 范围改用 new_k；Q 范围保持 tmp8；V 范围保持 tmp9。
tmp11 = tl.where(k_mask, tmp6, tmp10)

# 将完整的一行写回原始 QKV。
tl.store(in_out_ptr0 + (x0 + 4608*x1), tmp11, mask)
```

因此各通道实际发生的是：

```text
Q：从 new_q 读出，经 tmp8 → tmp10 → tmp11，写回 QKV 的 Q 区域
K：从 new_k 读出，经 tmp6 → tmp11，写回 QKV 的 K 区域
V：从原始 QKV 读出到 tmp9，再经 tmp10 → tmp11 原样写回
```

这就是该融合组的全部数据流。`tl.where` 只是从两个值中选择一个，不会改变被选中值的数值，所以这里确实没有需要 fp32 精度的计算。

#### fp32 提升在这条数据流中插入了什么

修复前，三次 load 后都额外追加了 `.to(tl.float32)`：

```python
tmp9 = tl.load(...).to(tl.float32)  # 内存中的 bf16 → kernel 内的 fp32
tmp8 = tl.load(...).to(tl.float32)
tmp6 = tl.load(...).to(tl.float32)

# tmp9/tmp8/tmp6 都是 fp32，where 也在 fp32 临时值上选择。
tmp10 = tl.where(..., tmp8, tmp9)
tmp11 = tl.where(..., tmp6, tmp10)

# 写回 bf16 buffer 时再降回 bf16。
tl.store(..., tmp11, ...)
```

也就是说，同一份数据经历了：

```text
内存中的 bf16
  → load 后转换为 fp32 临时值
  → 不做算术，只做 where 选择
  → store 时转换回 bf16
```

bf16 → fp32 → bf16 在这条路径上是无损往返，不会改善结果。它增加的是转换指令和 fp32 中间值的寄存器/片上缓冲占用，而不是提高计算精度。

这里的 **cast** 就是数据类型转换。它发生在 kernel 内部的临时值上，不等于另外创建一份完整 fp32 QKV 内存，所以收益不能只按外部内存流量估算，还会受到片上资源和 tile 选择的影响。

#### 为什么代码会默认进行这次提升

对乘加、sigmoid、归约等操作，把 bf16/fp16 提升为 fp32 可以降低舍入误差。Inductor 因而提供了 `codegen_upcast_to_fp32` 这一通用策略。

但融合组之间并不相同：

- Sigmoid+Mul 有真实算术，需要保留 fp32 计算路径；
- QKV 回写只有 load、where 和 store，可以安全地保持 bf16；
- 真实 dtype 转换、混合 dtype 或归约也不能按纯搬运处理。

所以目标不是全局关闭 fp32 提升，而是只让能够证明为纯搬运的 kernel 跳过它。

#### 第一次实现具体做了什么，为什么没有生效

第一次实现分三步：

1. 增加 `_npu_is_pure_data_movement(node_schedule)`，检查融合组的 FX origins，只有视图、切片、拷贝和选择等操作才返回 `True`。
2. 在调度生成 kernel 时设置：

   ```python
   kernel._npu_pure_data_movement = True
   ```

3. 修改 `npu_triton_compute_type(dtype)`，逻辑相当于：

   ```python
   if dtype in (torch.float16, torch.bfloat16):
       if 上游和 NPU 的 upcast 开关都开启 and not V.kernel._npu_pure_data_movement:
           return "tl.float32"
       return 原生 bf16/fp16 类型
   ```

这套修改的意图是“纯搬运 kernel 的计算类型保持 bf16”。单元测试也验证了：把 `V.kernel._npu_pure_data_movement` 设为 `True` 后，调用 `npu_triton_compute_type(torch.bfloat16)` 会返回 `tl.bfloat16`。

但实际生成的 QKV 仍有三处 `.to(tl.float32)`。原因是 load 的类型不是在 `npu_triton_compute_type()` 里决定的。

#### 两条 cast 路径到底有什么区别

第一条是刚刚修改的 **compute-type 路径**。某些代码在需要计算类型时会这样调用：

```text
compute 型 to_dtype / 部分常量定型
  → triton_compute_type(dtype)
  → NPU 替换后的 npu_triton_compute_type(dtype)
  → 根据纯搬运标记返回 fp32 或 bf16
```

第一次实现只改变了这条调用链。

第二条是 **load 级路径**。NPU 的 `NPUTritonKernel.load()` 会调用上游 `TritonKernel.load()`，上游在其中直接拼接字符串，逻辑相当于：

```python
line = "tl.load(...)"

if (
    dtype in (torch.float16, torch.bfloat16)
    and config.triton.codegen_upcast_to_fp32
):
    line += ".to(tl.float32)"
    dtype = torch.float32
```

这里的关键是：**这段代码没有调用 `triton_compute_type()`，也不会读取第一次实现添加到 `npu_triton_compute_type()` 中的纯搬运条件。**

所以两条路径的实际结果是：

| 生成位置 | 是否经过第一次修改的函数 | 第一次实现后的结果 |
|---|---|---|
| compute 型 `to_dtype` 等 compute-type 调用 | 是 | 直接函数测试可返回 bf16；没有真实 kernel 证明此分支是必要的 |
| `tl.load` 后直接追加的 cast | 否 | 仍然生成 `.to(tl.float32)` |

诊断时虽然 `_npu_is_pure_data_movement()` 已返回 `True`，但 compute/store 类型函数根本没有参与这三处 load cast 的生成。这证明问题不是标记设置太晚，而是第一次实现没有覆盖真正决定 load 类型的分支。

最终提交删除了 compute-type 中的纯搬运条件：它只有直接函数测试，没有真实 kernel 证据证明该分支是纯搬运场景所必需的。最终只保留纯搬运判定和已经验证生效的 load 级修改。

社区 `!48031` 处理的是另一个问题：让上游开关同时控制这两条路径中的 NPU compute-type 行为。它并没有根据 kernel 是否纯搬运作区分，因此不能直接替代本次修复。

#### 最终修复为什么能够生效

NPU 已经在 `NPUTritonKernel.load()` 中包装了上游 load。最终修复就在这个真实入口中，对纯搬运 kernel 临时关闭上游开关：

```python
if self._npu_pure_data_movement:
    with config.patch("triton.codegen_upcast_to_fp32", False):
        result_var = super().load(name, index)
else:
    result_var = super().load(name, index)
```

这样，上游执行到前面那段 `if dtype ... and codegen_upcast_to_fp32` 时，纯搬运 kernel 看到的开关是 `False`，便不会追加 `.to(tl.float32)`。`with` 结束后配置恢复，其他 kernel 仍使用原来的 fp32 策略。

这个 `config.patch` 发生在**编译生成代码期间**，不是 NPU 每次执行 kernel 时都做配置切换，因此不会给设备端增加运行时开关。

判定还要求浮点输入和输出 dtype 一致，避免把 bf16 → fp16 等真实类型转换误当成纯搬运。最终三处 load 的 fp32 转换全部消失，QKV 各形状数值差为 0；Sigmoid+Mul 等算术路径没有改变。

#### 实际收益

相对 `!50025 + clamp` 基线，load 级修复并重新 autotune 后：

| token 数 | 基线 | 修复后 | 时延降低 |
|---:|---:|---:|---:|
| 2048 | 33.826 us | 18.552 us | 45.2% |
| 512 | 10.8465 us | 6.1925 us | 42.9% |
| 128 | 4.832 us | 3.094 us | 36.0% |
| 8 | 4.1015 us | 2.104 us | 48.7% |

2048 token 的 autotune XBLOCK 同时从 16384 调整为 12128，因此这组数字是“去除 cast 后重新 autotune”的整体效果，不是固定 tile 下只测转换指令的收益。

### 4.4 QKV 全通道读写来自完整的 slice_scatter 语义

#### 模型源码只要求更新 Q/K

模型首先通过一次投影得到 `[token, 4608]` 的 QKV，再拆成三个视图：

```text
Q：[0, 3584)
K：[3584, 4096)
V：[4096, 4608)
```

旋转位置编码只生成新的 Q/K，随后源码执行：

```python
qkv[:, :3584].copy_(new_q)
qkv[:, 3584:4096].copy_(new_k)
```

**`copy_` 是 mutation（原地修改）**：目标只是 Q/K 两个视图，V 不应被访问。原生 `copy_` 是覆盖写，只需读取源数据并写入目标，不需要读取目标旧值。

#### Inductor 为什么会变成完整 slice_scatter

Inductor 为了用函数式图表达计算和依赖关系，会进行**函数化**：把“某个 buffer 被原地修改”转换成“根据旧值生成一个完整新值”。这样可以明确每个节点的输入、输出和依赖，便于后续调度与融合。

`slice_scatter` 的语义就是：生成完整输出，指定区间使用新值，其余区间保留旧值。两次局部 Q/K 更新因此变成类似：

```python
old = load(qkv)                    # 读取完整 4608 通道
out = where(q_range, new_q,
       where(k_range, new_k, old)) # V 使用 old
store(qkv, out)                    # 写回完整 4608 通道
```

**融合**把这个完整计算与其他搬运合并进一个 Triton kernel，减少 kernel 数量和中间操作，但也保留了完整旧值读取和完整写回。于是 V 的 512 个通道没有逻辑修改，却经历“读出旧值、原样写回”。

因此，全通道读写不是通道索引写错，也不是 V 被错误更新，而是当前完整 `slice_scatter` 表达带来的无效流量。

#### default 为什么没有这部分流量

default 没有执行这项融合优化，而是保留原生切片 `copy_`：

```text
目标视图 1：qkv[:, 0:3584]，只处理 Q
目标视图 2：qkv[:, 3584:4096]，只处理 K
```

虽然两个视图的底层 storage 都是完整 QKV，但原生算子的迭代域只是目标视图本身。V 不在任何目标视图内，所以既不读取，也不写回。

按每行元素数估算：

```text
当前融合形式：
  读旧 QKV 4608 + 读 new_q/new_k 4096 + 写完整 QKV 4608
  = 13312 个元素

只复制 Q/K：
  读 new_q/new_k 4096 + 写 Q/K 4096
  = 8192 个元素
```

2048 token、bf16 每元素 2 字节时，分别约为 `54.52 MB` 和 `33.55 MB`，差值约 `20.97 MB`。这是理论流量差，不等于相同比例的耗时收益。

#### 为什么不能直接退回 default 或只删 V 的 load

default 更省流量，但可能有更多原生 kernel 和启动开销。当前组合修复后的融合 QKV 为 `18.552 us`，default 模型路径相关 10 个 kernel 合计约 `69.86 us`，说明融合整体仍有价值，不能简单全面禁止。

同时，只删除 V 的 load 并不安全：当前形式仍有完整 store，V 位置需要旧值。如果只删 load，完整 store 可能把未定义值写进 V。

真正安全的优化需要证明：

- 最终写回与原始输入使用同一块有效存储，V 中已经保存正确的旧值；
- Q/K 更新区域内的输出完全不依赖旧值；
- 更新区域之外不再执行写回；
- 别名、依赖和后续读取关系保持不变。

然后同时缩小 load、store 和迭代域。返回新 tensor 的普通 `slice_scatter` 不能直接这样优化，因为新 buffer 的 V 区域仍必须初始化；只有能够证明原地复用的场景才可以保留 V。

这项优化涉及 mutation、别名分析、动态形状、grid、tile 和 autotune，不应针对 4608 或 QKV 名称做特判。当前先保留为后续图级优化空间。

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

判定还检查浮点输入 dtype 与输出是否一致，避免把 bf16 → fp16 等真实转换误判为纯搬运。最终单元测试 `8 passed`，覆盖 load 级开关、配置恢复、非纯搬运不跳过和混合输入 dtype 拒绝等情况。

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

最终验证使用 `install_pure`，精度和性能进程均确认加载该隔离目录，而不是主 conda。Qwen2.5-VL-7B-Instruct 和 Qwen3.8-27B-w8a8 的性能冒烟和两轮完整性能均已执行完成。删除未验证的 compute-type 修改后，QKV 最终生成代码与模型测试版本只有缓存路径差异，因此不影响这些结果。

固定输入的 eager/Inductor 输出均与各自 prefix 逐字节相同，说明修复没有引入额外输出变化。Qwen3.8-27B-w8a8 原有的 eager/Inductor 精度差异没有扩大。

以下数字是 Inductor 相对 default 的加速比，大于 1 表示 Inductor 更快，小于 1 表示仍慢于 default。每轮都重新测量 default 和 Inductor。表格直接列出两轮结果，不做平均，也不挑选表现较好的一轮。

### Qwen2.5-VL-7B-Instruct：两轮全部满足验收条件

| 负载 | 轮次 | 吞吐加速比 | TPOT 加速比 | TTFT 加速比 |
|---|---:|---:|---:|---:|
| ShareGPT QPS1 | 1 | 1.000 | 1.016 | 0.990 |
| ShareGPT QPS1 | 2 | 1.000 | 1.021 | 0.988 |
| ShareGPT QPS4 | 1 | 1.001 | 1.043 | 0.997 |
| ShareGPT QPS4 | 2 | 1.001 | 1.045 | 0.982 |
| ShareGPT QPS16 | 1 | 1.035 | 1.042 | 1.190 |
| ShareGPT QPS16 | 2 | 1.036 | 1.042 | 1.196 |
| ShareGPT QPSinf | 1 | 1.046 | 1.048 | 1.044 |
| ShareGPT QPSinf | 2 | 1.022 | 1.022 | 1.001 |
| 随机 8K/1024 | 1 | 1.029 | 1.034 | 0.982 |
| 随机 8K/1024 | 2 | 1.029 | 1.033 | 0.995 |
| 随机 32K/1024 | 1 | 1.015 | 1.023 | 0.983 |
| 随机 32K/1024 | 2 | 1.017 | 1.021 | 0.998 |

32K 相对 prefix 的两轮汇总结果为：吞吐由 `0.924` 提高到 `1.016`，TPOT 由 `0.938` 提高到 `1.022`，TTFT 由 `0.864` 提高到两轮分别 `0.983`、`0.998`。32K 吞吐和 TTFT 均提高，8K 不下降，其余负载的下降不超过 2%，全部满足验收条件。

TTFT 的改善不代表已经完全追平 default。prefix 两轮 Inductor 分别约慢 `0.75 s`，最终两轮分别约慢 `83 ms` 和 `9 ms`；最终两轮 TTFT 加速比仍分别为 `0.983`、`0.998`。

### Qwen3.8-27B-w8a8：长输入改善，QPS1 TTFT 未通过

| 负载 | 轮次 | 吞吐加速比 | TPOT 加速比 | TTFT 加速比 |
|---|---:|---:|---:|---:|
| ShareGPT QPS1 | 1 | 1.000 | 1.014 | 1.013 |
| ShareGPT QPS1 | 2 | 1.000 | 1.009 | 0.960 |
| ShareGPT QPS4 | 1 | 1.001 | 1.034 | 1.231 |
| ShareGPT QPS4 | 2 | 1.001 | 1.005 | 1.018 |
| ShareGPT QPS16 | 1 | 1.030 | 1.029 | 1.045 |
| ShareGPT QPS16 | 2 | 1.004 | 1.003 | 1.005 |
| ShareGPT QPSinf | 1 | 1.016 | 1.016 | 1.016 |
| ShareGPT QPSinf | 2 | 1.012 | 0.998 | 1.015 |
| 随机 8K/1024 | 1 | 1.010 | 1.016 | 0.976 |
| 随机 8K/1024 | 2 | 1.010 | 1.012 | 1.003 |
| 随机 32K/1024 | 1 | 1.009 | 1.019 | 0.976 |
| 随机 32K/1024 | 2 | 1.010 | 1.015 | 0.990 |

32K 相对 prefix 的两轮汇总结果为：吞吐由 `0.955` 提高到 `1.009`，TPOT 由 `0.967` 提高到 `1.017`，TTFT 由 `0.914` 提高到两轮分别 `0.976`、`0.990`，三项均明显改善。两轮 TTFT 仍分别表示 Inductor 比 default 慢约 2.4% 和 1.0%。

未通过项是 ShareGPT QPS1 的 TTFT：第 1 轮为 `1.013`，第 2 轮降至 `0.960`，两轮方向相反。按既定两轮汇总规则，prefix 为 `1.024`，最终为 `0.986`，下降 `0.038`，超过允许的 `0.02`，因此 Qwen3.8-27B-w8a8 不能判定为全部通过。直接展示两轮也可以看到，该负载存在明显波动，后续需要 Profile 区分真实回落与测量波动。

结果目录分别为 `results/vl_vl_pure_final` 和 `results/qwen27b_qwen27b_pure_final`，比较使用 `compare_perf.py --repeats 2`。`PERF_full2_RC=0` 只表示测试执行成功，不等于验收通过。

## 8. 交付物

| 内容 | 位置 |
|---|---|
| 静态启动器、clamp、纯搬运补丁 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/patches/` |
| 纯搬运根因与结果摘要 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/PURE_DATA_MOVEMENT_RESULT.md` |
| 定位与实验脚本 | `server-scripts/benchmark-diagnosis/20261126_triton_kernel_perf/` |
| kernel 原始结果 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/server/locate/k2_zeros/results_sizes/` |
| 模型精度、性能结果和日志 | `server-results/benchmark-diagnosis/20261126_triton_kernel_perf/server/results/`、`server-results/benchmark-diagnosis/20261126_triton_kernel_perf/logs/` |
| 工作过程记录 | `server-scripts/benchmark-diagnosis/20261126_triton_kernel_perf/WORKLOG.md` |

测试汇总：静态启动器 `84 passed`，fast launcher `19 passed`，clamp 相关 pytest 入口 `2 passed`，纯搬运 `8 passed`。四个目标 kernel 的数值检查均通过。

## 9. 未解决的问题

1. **Qwen3.8-27B-w8a8 的 ShareGPT QPS1 TTFT 未满足验收条件。** 两轮汇总值由 prefix 的 `1.024` 降至 `0.986`，回落超过 2%；当轮结果分别为 `1.013`、`0.960`。此外，Qwen2.5-VL-7B-Instruct 的最终 32K TTFT 两轮为 `0.983`、`0.998`，Qwen3.8-27B-w8a8 为 `0.976`、`0.990`，说明 Inductor 尚未完全追平 default。剩余差距需要结合新的 prefill Profile 定位，不能仅根据目标 kernel 推断。
2. **GDN 在中等形状上仍慢于原生拷贝。** 2048 token 当前为 `13.4185 us`，原生为 `9.2705 us`。GDN 也未命中当前纯搬运判定，生成代码仍保留 fp32 转换；是否需要调整融合组判定或 tile 候选，留待后续分析。
3. **QKV 仍读写全部 4608 通道。** 2048 token 时当前总流量约 `54.52 MB`，只复制 Q/K 的形式约 `33.55 MB`。该优化涉及 mutation、别名、迭代域和调度，按当前任务决定暂缓。
4. **其他覆盖范围未扩展。** `cpp_wrapper` 和 AscendForge 门禁未在本次验证中运行，不将其报告为通过。

## 修订记录

- 2026-11-27：将静态启动器方案更新为社区 `!50025`；补充 clamp 的社区背景与验证；加入纯搬运 load 级根因、修复和四形状收益；更新四个 kernel 的最终结果、模型精度状态及后续未解决项。
- 2026-11-27：扩展第 4 节，补充 launcher、AutoBlockify、逻辑/物理 grid、lane/mask、仿射地址、fp32 提升、函数化和 `slice_scatter` 的解释，并明确 QKV 各修复组合的耗时关系。
- 2026-11-27：进一步补充第 4 节的端到端代码生成与启动链路、AutoBlockify 的引入过程、实际地址和通道示例、原设计动机、各方案的影响范围及独立对照数据。
- 2026-11-27：重写第 4.3 节，逐变量说明 QKV 的内存读写和 dtype 变化，完整展示第一次实现的判定与 compute-type 修改，并用调用链和代码分支解释 load 级修复为何必要。
- 2026-11-27：合并纯搬运的两个本地提交，删除缺少真实 kernel 证据的 compute-type 修改，保留 load 级修复；更新为 8 条单元测试及最终生成代码验证。
- 2026-11-27：补充两个模型的两轮性能结果：Qwen2.5-VL-7B-Instruct 全部满足验收条件；Qwen3.8-27B-w8a8 长输入改善但 QPS1 TTFT 回落超过 2%，并明确两者的 32K TTFT 仍略慢于 default。
- 2026-11-27：模型性能表改为逐项展示两轮加速比，不再只显示平均后的结果；模型名称统一为完整名称。

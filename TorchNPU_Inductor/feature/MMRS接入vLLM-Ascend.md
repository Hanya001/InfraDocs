# Matmul + ReduceScatter（MMRS）接入 vLLM Ascend Inductor

## 1. 阶段状态

- 日期：2026-11-22。
- 服务器：A5-25，Ascend950DT，CANN 9.2.0。
- 容器：`inductor_feature_z00896543_20261122_140509`。
- 分支：`hanya/feat/inductor-low-cost-fusions`。
- 实现目标：复用已有 `MatmulReduceScatterFusionPass` 和 `torch_npu.npu_mm_reduce_scatter_base`，使 Inductor 路径在 SP 场景下融合非量化 GEMM 与 reduce-scatter，不开发新 NPU kernel。大模型实测为负收益，最终通过 `fuse_mm_reduce_scatter=False` 默认关闭，仅允许显式实验开启。

阶段提交：

| 内容 | Commit | Patch |
|---|---|---|
| 将已有 MMRS pass 注入 Inductor 路径 | `96d27159cafd0d162ba9307bc930fc511df6a062` | `Patch/MMRS/0003-chain-mmrs-fusion-into-inductor.patch` |
| 支持 Ascend 自定义 SP reduce-scatter 图形态 | `88a4a9b31e11806e01c8559605c2cf3a6f0a85e1` | `Patch/MMRS/0005-support-ascend-sp-in-mmrs-fusion.patch` |
| 支持动态 token 维度的 fake/meta 形状 | `0f46d8c27b7ce3e637f44d216e3c3bdcba73b12c` | `Patch/MMRS/0006-support-symbolic-shapes-in-mmrs-fusion.patch` |
| 保留 Ascend SP 的动态 padding 语义 | `5ef8650782885faefe5ba5d95f8323d5da1ba888` | `Patch/MMRS/0007-preserve-sp-padding-in-mmrs-fusion.patch` |
| 整理 MMRS/VF 相关格式，未混入其他文件 | `c8af34cae813691eecba0681cd96fab7b0686240` | `Patch/MMRS/0008-format-mmrs-and-vf-changes.patch` |
| 实测负收益后将 MMRS 改为默认关闭的 opt-in 能力 | `2c59fe8a99e08df9c27f0f83b4d625ed6ed6cae1` | `Patch/MMRS/0009-make-mmrs-fusion-opt-in.patch` |

## 2. 原缺口

旧 `GraphFusionPassManager` 已在以下条件下注册 MMRS：

- `enable_sp(config)` 为真；
- 当前硬件支持 `GRAPH_MM_REDUCE_SCATTER_FUSION`。

A5 已声明支持该能力。Inductor 使用的 `AscendPostGradPassManager` 原先只注入 norm-quant、qknorm-rope、muls-add 和 elide，没有 MMRS。第一阶段只复用旧条件和已有 pass，将其补入 Inductor 路径。

已有 pass 支持：

- `torch.ops.vllm.unquantized_gemm`；
- BF16/FP16；
- TP 2、4、8；
- 无 bias；
- 普通 `vllm.reduce_scatter` 以及显式 cat/pad 形式。

## 3. 实际图形态适配

Qwen3.5/Qwen3.8 dense SP 不使用普通 `vllm.reduce_scatter`，而是：

```text
unquantized_gemm -> vllm.ascend_sp_reduce_scatter_impl
```

因此新增 `AscendSPMatmulReduceScatterPattern`，继续复用同一个 MMRS replacement。标准 reduce-scatter 的 dim、world_size 和 group_name 检查只应用于标准算子；Ascend SP 自定义算子的 TP 信息由 pass 初始化时的 TP group 提供。

单元测试覆盖了 SP 与非 SP、硬件支持与不支持、普通 MMRS 和 Ascend SP MMRS。

## 4. 动态 token 维度问题与修复

首次真实 TP2 编译已经命中 MMRS，但 torch_npu 的 meta 实现使用：

```python
math.floor(self.size(0) / world_size)
```

该实现不能处理 vLLM 动态 token 维度，编译报 `AssertionError: -2`。

为不改变运行时算子，新增 vLLM 自定义算子薄包装：

- 实际执行仍调用 `DeviceOperator.npu_mm_reduce_scatter_base`；
- fake/meta 使用 `x1.shape[0] // world_size` 和 `x2.shape[1]` 构造输出；
- 保留符号行数，支持动态 shape 和后续 ACL Graph 编译。

新增符号行数回归测试，防止以后直接调用无法处理符号维度的 torch_npu meta。

## 5. Ascend SP padding 问题与修复

第二次真实运行进入 MMRS 后，运行时算子报：

```text
The M-axis in input of Matmul should be divisible by world_size.
```

原因是 Ascend SP 自定义算子原本会在内部把 token 行数补齐到 TP 倍数。直接换成 MMRS 后，这一语义必须保留。

Ascend SP replacement 现在先计算：

```python
pad_rows = (-x.shape[0]) % tp_size
```

再对 GEMM 输入 padding 后调用 MMRS。0 行 padding 时，`constant_pad_nd` 自然退化为原输入；非 0 行时，保证 M 能被 TP 整除。单元测试同时覆盖 7 行补 1 行和 8 行不补的情况，以及输出 shape。

## 6. 单元测试

最终相关测试：`250 passed`，包括：

- 默认不注入 MMRS、显式开启后按 SP 和硬件条件注入；
- 普通和 Ascend SP pattern；
- 不支持 dtype、bias、world size 和错误 group 的拒绝；
- 普通 cat/pad 形式；
- 动态符号行数；
- Ascend SP 动态 padding；
- 既有 Inductor 配置和 AscendConfig 回归。

Ruff 自动修复只保留在 MMRS/VF 相关文件中，其他被工具触及的文件均已恢复。格式调整单独提交为 `c8af34cae`，不混入功能 patch。

## 7. 真实模型验证与性能

### 7.1 Qwen3.5-0.8B TP2 冒烟

配置：TP2、`enable_flashcomm1=True`、Inductor、4 条提示、最多生成 32 token、一次 warmup 后执行一轮。

结果：

- 启动时间：151.263 s；
- 生成时间：1.012 s；
- 输出非空且内容正常；
- 输出 SHA256：`5a08ced95d3e333ed56aebaa64c3e4e622fd23f14cb78cf43f299a542fbcc359`；
- 生成代码实际调用 `torch.ops.vllm.npu_mm_reduce_scatter_base.default`；
- `fx_graph_transformed` 中保留 `PythonMod(-s18, 2)` 动态 padding 和符号输出形状。

这证明 pass 注册、真实 pattern 命中、动态 meta、动态 padding 和实际算子执行均已打通，不只是配置或单元测试成立。

### 7.2 Qwen3.8-27B A/B

配置：TP2、FlashComm1 SP、BF16、4 条约 4000 字符的提示，每条最多生成 64 token，关闭 prefix caching；一次 warmup 后测量三轮，编译缓存按组隔离。

开启组已完成：

- 启动时间：282.434 s；
- 三轮长 prefill + decode：2.433247、2.431365、2.430447 s，中位数 2.431365 s；
- 三轮输出一致，首轮输出 SHA256：`4b8ca12baa4499719db1fef0bb2dbe8ed3f4d0cf18dce77f78f83c7edb7e9553`；
- 5 个实际生成子图的 `output_code.py` 均包含两次 `torch.ops.vllm.npu_mm_reduce_scatter_base` 调用；
- 对应 `fx_graph_transformed.py` 保留 `PythonMod(-s18, 2)` 和动态 padding，证明 27B 路径与小模型使用同一有效实现。

对照源码固定在接入前的 `7fc0fed88421587b65b1275920b0a2273c045f95`。旧源码树第一次运行还缺少当前树中已构建的扩展，补齐相同构建产物后，后续三次尝试均在 MMRS 注册之前的 `AddRMSNormQuantPattern` 注册阶段报 `aclnnAddRmsNormBias`/`libopapi.so` 缺失。为隔离这个与 MMRS 无关的问题，最终对照组设置 `fuse_norm_quant=False`。

该处理不会从本次 BF16 实际计算图中删除 norm-quant 融合：已逐个检查开启组全部 6 个 `output_code.py`，`rms_norm_quant` 出现次数为 0。三次失败日志和最终对照日志均保留，不把失败启动当作性能数据。

有效 A/B 结果：

| 指标 | 未接入 MMRS | 开启 MMRS | 差异 |
|---|---:|---:|---:|
| 启动时间 | 276.649 s | 282.434 s | 开启增加 5.785 s（2.09%） |
| 第 1 轮生成 | 1.811025 s | 2.433247 s | — |
| 第 2 轮生成 | 1.808051 s | 2.431365 s | — |
| 第 3 轮生成 | 1.808825 s | 2.430447 s | — |
| 三轮生成中位数 | 1.808825 s | 2.431365 s | 开启延迟增加 34.42% |
| 输出 SHA256 | `4b8ca12baa4499719db1fef0bb2dbe8ed3f4d0cf18dce77f78f83c7edb7e9553` | 相同 | 逐 token 文本一致 |

未接入组的 6 个 `output_code.py` 中，`npu_mm_reduce_scatter_base` 出现次数为 0，其中 5 个保留 `ascend_sp_reduce_scatter_impl`；开启组对应 5 个文件改为实际 MMRS 调用。因此这组差异确实发生在 MMRS 替换后，不是“开关开了但图没变”。

**收益结论：当前 MMRS 实现在本模型和负载下是明确负收益。** 三轮生成中位数增加 34.42%，等效吞吐下降约 25.60%；单次冷启动也略慢，不能把启动波动解释为优化。可能原因包括融合算子在这些 shape 上的实现效率、动态 padding 和图拆分变化，但本阶段没有 profiler 证据，不进一步断定具体算子内部原因。

基于实测，Inductor 路径不应默认启用 MMRS；保留代码作为显式 opt-in 的实验能力，默认配置继续使用原生 SP reduce-scatter。

## 8. 原始结果与回退

原始结果归档：

```text
server-results/inductor-feature/20261122_140509/mmrs_qwen08b/
server-results/inductor-feature/20261122_140509/mmrs_qwen27b/
```

按提交顺序应用 `Patch/MMRS/0003`、`Patch/MMRS/0005`、`Patch/MMRS/0006`、`Patch/MMRS/0007`、`Patch/MMRS/0008`、`Patch/MMRS/0009` patch。默认配置不启用 MMRS；如确需复现实验，使用：

```python
additional_config={
    "ascend_compilation_config": {
        "fuse_mm_reduce_scatter": True,
    }
}
```

关闭时删除该配置或设为 `False` 即可，不需要回退动态 shape 和 padding 修复。`Patch/MMRS/0009` 之前的提交会在满足 SP 和硬件条件时自动注入 MMRS，不建议直接作为默认部署版本。

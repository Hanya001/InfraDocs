# Qwen3-0.6B：原生 `npu_rms_norm` 与 Inductor Triton RMSNorm 数值差异定位

## 1. 问题状态与结论

- **状态**：根因已定位，尚未提交生产修复。
- **直接原因**：A5 上默认路径/eager 使用原生 `npu_rms_norm`，Inductor 则将其分解为 FP32 计算并生成 Triton kernel。两种实现的平方和归约顺序及 `rsqrt` 计算方式不同，产生 BF16 末位级差异，并在贪心解码的近似平票位置翻转 argmax。
- **性质判断**：这是不同数值实现之间的确定性差异，不是随机采样、批量调度、权重加载或 KV cache 整体损坏。现有证据也不能证明 Inductor 使用了错误公式或存在系统性精度下降。
- **精确归因边界**：端到端 A/B 已闭环到 RMSNorm 实现差异；归约与 `rsqrt` 的 FP32 末位级差异也已分别测量。1024 维直接 CANN 诊断 kernel 与原生预编译路径有 3/16 样本相差 1 FP32 ULP，因此不声称完全复现其所有调度和编译细节。

本文是追踪文档 O-02 通用 near-tie 问题的具体闭环案例。W-01 讨论量化场景的 Norm+Quant 融合，算子链和实验对象不同，不覆盖本文问题。

## 2. 环境与复现口径

实验使用 A5/Ascend950DT，主要组件为：

- CANN 9.2.0、torch 2.13.0+cpu、torch-npu 2.13.0+gitc798c65。
- triton-ascend 3.6.0.post2、vLLM 0.27.1、vllm-ascend 0.27.1rc0。
- 模型 `/mnt/weight/Qwen/Qwen3-0.6B`，BF16，单卡，温度 0，随机种子 0，关闭前缀缓存。
- 默认路径使用 AscendCompiler，对照路径使用 `compilation_config={"backend":"inductor"}`，eager 参考使用 `enforce_eager=True`。

原始模测报告见 [A5 RESULTS.md](../../../server-scripts/bench213/20261112_A5机器模测/20261112_6models_01/RESULTS.md)。

注意：测试脚本将原始字符串直接传给 `llm.generate()`，没有应用 chat template。因此，输出中的重复问题或回答质量差不能单独用于判断模型精度；本文只讨论编译路径之间的确定性数值一致性。

## 3. 原始现象

12 条提示词中，默认路径与 Inductor 有 10 条完整 128 token 一致，分歧为：

| 提示词编号 | 首个不同 token 位置，从 0 计数 | 默认路径 | Inductor |
|---:|---:|---|---|
| 3：光速 | 11 | `3×10^8米每秒` | `$3 \times 10^8$ 米每秒` |
| 7：2 的 10 次方 | 5 | `2^10等于多少` | `2^10 的值是多少` |

两条路径各自重复两次生成均逐 token 稳定，排除了随机采样导致分歧的可能。分歧从中间 token 开始，两路都完整生成 128 token，不属于首 token 即损坏的严重生成故障。

原始比对见 [compare.json](../../../server-scripts/bench213/20261112_A5机器模测/20261112_6models_01/Qwen3-0.6B/compare.json)。

## 4. 实验设计与结果

### 4.1 分歧点 logprobs

| 位置 | 默认路径 | 常规 Inductor | 判断 |
|---|---|---|---|
| 编号 3，token 11 | `3`=-1.537523；` $`=-1.662523 | ` $`=-1.503707；`3`=-1.628707 | 两路均只相差 0.125 logprob，但 Top1/Top2 顺序翻转 |
| 编号 7，token 5 | `等于`=-1.357794；` 的`=-1.482794 | 两者均为 -1.458127 | Inductor 中两个候选在已保存精度下并列 |

常规 Inductor 的 12 条输出虽然仍有 10 条 token 一致，但前 16 个位置的完整 Top-5 logprobs 与 eager 没有任何一条完全相同，说明微小数值差异普遍存在，只在近似平票时才暴露为文本分歧。

### 4.2 单请求与批量对照

对编号 3、7 分别运行单请求和 12 请求批量，默认、Inductor、原生 RMSNorm A/B 三组实验中单请求均与批量输出一致。因此 batch size 和调度差异不是本次触发条件。

### 4.3 保留原生 RMSNorm 的端到端 A/B

诊断实验仅取消 Inductor 的 `npu_rms_norm` 分解，使其保留原生 NPU 实现，其余 Inductor 编译配置不变。取消分解的补丁同时应用于父进程和 spawn worker，并通过生成图确认实际执行的是 `torch.ops.npu.npu_rms_norm`。

| 路径 | 与 eager 完整 128 token 一致 | 与 eager 的 logprobs 对比 |
|---|---:|---|
| 默认路径 | 12/12 | 前 16 token 的完整 Top-5 logprobs 12/12 完全相同 |
| 常规 Inductor | 10/12 | 前 16 token 的完整 Top-5 logprobs 0/12 完全相同 |
| Inductor + 原生 RMSNorm | **12/12** | **完整 128 token 的 Top-5 logprobs 12/12 完全相同** |

另外，Inductor + 原生 RMSNorm 与原默认路径的完整 128 token 也是 12/12 完全一致。该对照直接确认 RMSNorm 实现差异是两处分歧的来源。

常规 Inductor 相对 eager 的 Top-1 logprob 差异，在只包含相同上下文前缀的 178 个可比位置上，平均绝对差为 0.0149896，最大绝对差为 0.100333。

实验脚本见 [probe_logprobs.py](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/probe_logprobs.py)，完整 A/B 输出见 [原生 RMSNorm results_full/batch12.json](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/inductor_native_rmsnorm/results_full/batch12.json)，eager 输出见 [eager/results_full/batch12.json](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/eager/results_full/batch12.json)。

### 4.4 RMSNorm 微基准

对输入 RMSNorm 和 Q/K RMSNorm 的六种形状，比较原生 NPU、Inductor/Triton 和 FP32 eager 参考：

```text
(12,1024)、(1,1024)、(12,16,128)、
(12,8,128)、(1,16,128)、(1,8,128)
```

16 组随机种子合计比较 851,968 个元素，原生与 Inductor 仅有 2 个元素不同，属于 BF16 末位量级。不同样本中，既存在原生实现更接近 FP32 的情况，也存在 Inductor 更接近 FP32 的情况，没有观察到 Inductor 系统性偏向更差精度。

微基准脚本及结果：

- [probe_rmsnorm.py](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/probe_rmsnorm.py)
- [probe_rmsnorm_sweep.py](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/probe_rmsnorm_sweep.py)
- [单组结果](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/rmsnorm/results/results.json)
- [16 组种子扫描结果](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/rmsnorm_sweep/results/results.json)

### 4.5 单独验证 `rsqrt` 的实现误差

为排除归约影响，构造每一行都为相同 BF16 数值的输入。BF16 常数转 FP32 后的平方，以及固定数量相同平方值的求和和均值，在该实验中均保持精确。因此，原生与 Inductor 的 `rstd` 差异只能来自后续倒数均方根计算。

262,144 个精确归约校准结果中：

| 指标 | 结果 |
|---|---:|
| 原生与 Triton 的 FP32 `rstd` 不同 | 73,328 个，约 27.97% |
| 两者最大差异 | 1 个 FP32 ULP |
| 原生相对 FP64 的最大相对误差 | 7.04×10⁻⁸ |
| Triton 相对 FP64 的最大相对误差 | 1.21×10⁻⁷ |
| 两者最终 BF16 输出 `y` 不同 | 0 个 |

这直接确认原生 Newton–Raphson 与 Triton `math.rsqrt` 会产生 FP32 末位级差异，但该差异不一定立即造成 BF16 输出变化。

随机输入下，归约与 `rsqrt` 叠加后的 `rstd` 最大相差 2 个 FP32 ULP；相对 FP64 参考，原生最大相对误差约 1.02×10⁻⁷，Triton 约 1.63×10⁻⁷，未观察到异常放大。

脚本和原始结果：

- [probe_rmsnorm_rstd.py](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/probe_rmsnorm_rstd.py)
- [rstd 分离实验结果](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/rmsnorm_rstd/results/results.json)

### 4.6 直接 CANN 归约诊断

为避免把 CPU 模拟当作最终证据，另行编译独立 Ascend C kernel，直接复用原生实现的 `CalculateSquareReduceSum` 和 `ComputeRstdNewtonRaphson`，导出 FP32 `sum(x²)` 和 `rstd`。最终实验采用每样本、每进程、单 block、单次计算，避免多 block 调度和 buffer 复用影响诊断。

每种维度使用 16 个独立随机样本：

| 指标 | 128 维 | 1024 维 |
|---|---:|---:|
| 原生平方和 vs FP64 最大误差 | 1 FP32 ULP | 1 FP32 ULP |
| Triton 平方和 vs FP64 最大误差 | 1 FP32 ULP | 1 FP32 ULP |
| 原生 vs Triton 平方和最大差异 | **2 FP32 ULP** | **2 FP32 ULP** |
| 原生 vs Triton `rstd` 最大差异 | **1 FP32 ULP** | **1 FP32 ULP** |
| 平方和最大相对差异 | 1.30×10⁻⁷ | 1.20×10⁻⁷ |

诊断 kernel 与现有原生算子的 `rstd` 校验结果：

- **128 维**：16/16 样本完全相同。
- **1024 维**：13/16 样本相同，另外 3 个样本相差 1 FP32 ULP。这说明该诊断 kernel 在 1024 维上并非与原生预编译路径完全同源同二进制；结论只用于确认误差实现和量级，不声称完全复现其所有调度细节。

这组实验直接确认：两种实现的归约和 `rsqrt` 均存在正常 FP32 末位级差异，误差足以改变近似平票处的 argmax，但不构成算法错误或异常精度下降的证据。

源码、构建和对照结果：

- [CANN 诊断 kernel](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/cann_diag/rms_norm_diag.cpp)
- [直接 CANN 对照脚本](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/cann_diag/compare_cann_diag.py)
- [直接 CANN 对照结果](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/cann_diag/results/results.json)

## 5. 对应实现及区别

### 5.1 默认/eager：原生 NPU 实现

调用链为：

```text
torch_npu.npu_rms_norm
  → torch.ops.npu.npu_rms_norm
  → torch_npu/csrc/aten/CustomRegisterSchema.cpp:11044
  → CANN aclnnRmsNorm
  → RmsNorm_normal_all_bf16_high_performance.o
  → KernelRmsNormRegBasePerf
```

A5 BF16 元数据包含 5000 和 2001 两个 kernel；本次 128/1024 维满足 full-load 条件，走 5000 分支。`high_performance` 入口将 5000 映射到 `KernelRmsNormRegBasePerf`。

主要计算为：

```cpp
CalculateSquareReduceSum<DX>(...);
ComputeRstdNewtonRaphson<false, true>(...);
ComputeY(...);
```

实现要点：

- BF16 输入和 gamma 转 FP32。
- 通过 `CalculateSquareReduceSum` 按向量宽度折叠归约平方和。
- 使用 Newton–Raphson 迭代及修正计算倒数均方根。
- 按 `(x * rstd) * gamma` 计算输出。
- FP32 转 BF16 显式使用 `CAST_RINT`。
- 第二个输出 rstd 为 FP32。

现场及对应源码文件：

- [torch_npu 注册声明](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/npu_native_functions_by_codegen.yaml)
- [aclnnRmsNorm 接口](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/aarch64-linux/include/aclnnop/aclnn_rms_norm.h)
- [A5 BF16 kernel 元数据](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/rms_norm/RmsNorm_normal_all_bf16_high_performance.json)
- [high_performance 入口](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/ops-nn/rms_norm_apt.cpp)
- [RegBasePerf 主实现](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/ops-nn/arch35/rms_norm_regbase_perf.h)
- [Newton–Raphson 实现](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/ops-nn/norm_common/reduce_common_regbase_part1.h)
- [折叠归约及 BF16 转换](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/native/ops-nn/norm_common/reduce_common_regbase_part2.h)

现场 CANN 只提供编译后的 `.o`。以上设备实现按 CANN v9.2.0 对应开源代码解读，入口、kernel 名称、dtype 和 `high_performance` 元数据与现场一致；官方源码见 [CANN ops-nn v9.2.0 RMSNorm](https://gitcode.com/CANN/ops-nn/tree/v9.2.0/norm/rms_norm)。

### 5.2 Inductor：分解后的 Triton 实现

TE 分解函数将原生算子替换为：

```python
dtype = x.dtype
x = x.float()
rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
output = (x * rsqrt * weight).to(dtype)
return output, rsqrt
```

分解位置见 [decomposition.py](../../../server-scripts/bench213/20261112_A5机器模测/accuracy_probe_qwen3_0_6b/source/inductor/decomposition.py)。

生成 kernel 的核心步骤为：

```python
x = tl.load(...).to(tl.float32)
square_sum = tl.sum(x * x, reduction_dim)
mean = square_sum / hidden_size
rstd = tl.rsqrt(mean + 1e-6)
y = x * rstd * weight
tl.store(..., y)
```

现场生成代码：

- [1024 维输入 RMSNorm](../../../server-scripts/bench213/20261112_A5机器模测/20261112_6models_01/Qwen3-0.6B/inductor/runtime/log/torch_compile_debug/run_2026_11_12_09_28_05_072230-pid_265878/torchinductor/model__0_inference_0.0/output_code.py)
- [128 维 Q/K RMSNorm](../../../server-scripts/bench213/20261112_A5机器模测/20261112_6models_01/Qwen3-0.6B/inductor/runtime/log/torch_compile_debug/run_2026_11_12_09_28_05_072230-pid_265878/torchinductor/model__1_inference_1.1/output_code.py)
- [TTIR：math.rsqrt 与 arith.truncf](../../../server-scripts/bench213/20261112_A5机器模测/20261112_6models_01/Qwen3-0.6B/inductor/runtime/log/vllm_cache/torch_compile_cache/2770f1d46f/rank_0_0/triton_cache/E4FCGIBJXBLLZJTVTT5XE45XY42XGKWX3P4DUC53OBC252AGILLA/triton_unk_fused_npu_rms_norm_1.ttir)

### 5.3 逐项对照

| 项目 | 原生 NPU | Inductor/Triton | 数值影响 |
|---|---|---|---|
| 输入及 gamma | BF16 转 FP32 | BF16 转 FP32 | 基本一致 |
| 平方和 | 向量宽度折叠归约 | `tl.sum`，降低为 `tt.reduce/linalg.reduce` | **归约顺序不同，可产生 FP32 舍入差** |
| 求均值 | 预计算 `avgFactor=1/N`，再乘平方和 | 平方和直接除以 N | 本次 N=128/1024，基本等价 |
| 倒数均方根 | Newton–Raphson 迭代及修正 | `math.rsqrt` | **近似、修正和舍入路径不同** |
| 输出计算 | `(x * rstd) * gamma` | `(x * rstd) * gamma` | 乘法顺序一致 |
| 第二个输出 rstd | FP32 | FP32 | 一致 |
| BF16 输出 | 显式 `CAST_RINT` | `arith.truncf` | 实现入口不同，但不是当前已证明的首要差异点 |
| 输入复用 | 复用 `xLocal` | reduction 和输出阶段分别读取 | 性能差异，不改变数学公式 |

## 6. 根因解释与处理建议

两条实现都计算：

```text
y = x × rsqrt(mean(x²) + epsilon) × gamma
```

但 FP32 加法不满足结合律，不同归约顺序会产生微小差异；`rsqrt` 的不同算法及舍入也会改变中间结果。差异经多层网络传播后改变 logits，并在 Top1/Top2 足够接近时改变贪心解码结果。

因此应将“直接触发点”和“数值实现根因”区分：

- **直接触发点**：编号 3 的 token 11、编号 7 的 token 5 发生 argmax 翻转。
- **数值来源**：原生 RMSNorm 与 Inductor 分解 RMSNorm 的归约和 `rsqrt` 实现不同。
- **尚未证明**：不能仅凭 token 不一致认定 Inductor 模型质量或数值精度不合格，也不能把本次结论直接外推到 O-02 的其他 MoE、量化或 PCP 用例。

处理建议：

1. 若验收要求与默认/eager **逐 token 完全一致**，可让 Qwen3 的 `npu_rms_norm` 保留原生 NPU 实现，不为它启用当前分解。该方向已有端到端 A/B 证据，但尚需形成正式配置或代码修改，并重新评估性能。
2. 若验收关注模型质量，应增加标准数据集和 logits 容差判据，不能只以 token 完全相等作为精度结论。
3. 归约和 `rsqrt` 的独立误差已通过直接 CANN/Triton 实验量化；若后续还需缩小到具体机器指令，可继续比较 BiShengIR 和最终设备代码，但这不是当前数值结论的前置条件。

当前仅新增诊断实验和文档，没有生产代码修复提交。





# 对于这种情况 社区通常怎么评价换backend精度是否可接受？

社区通常**不要求换 backend 后逐 bit 一致，也不单凭生成文本是否完全相同判断精度**。一般分三层评价：**算子数值误差、模型输出分布、下游任务指标**。

## 1. 算子层：与参考实现比较误差

常见做法是将 backend 输出与 eager、高精度 FP64 参考结果比较，使用：

- `torch.testing.assert_close` 的 `rtol/atol`；
- 最大绝对误差、最大相对误差；
- ULP 差异；
- 必要时检查均值偏差，排除系统性偏移。

容差根据 dtype、算子性质和输入范围确定，**没有适用于所有模型的统一阈值**。对于 reduction、`rsqrt`、transcendental 等运算，少量 FP32 ULP 差异通常被视为正常的浮点实现差异，而不是天然的功能错误。

同时还要求：

- 无异常 `NaN/Inf`；
- 各种 shape、dtype、边界输入均正确；
- 没有明显的系统性偏大或偏小；
- 不出现远超正常舍入范围的误差。

## 2. 模型层：比较 logits 和输出分布，而非只看 token

通常会固定同一批输入，对比 eager 与新 backend 的：

- logits 的绝对/相对误差；
- Top-k 重合度；
- 概率分布差异；
- greedy token 一致率；
- 不同输入长度、batch、shape 下的稳定性。

其中，**token 一致率更适合作为诊断指标，而不是唯一验收标准**。因为 greedy decoding 在近似平票时，即使 logits 只有正常浮点误差，也可能翻转 `argmax`，随后上下文不同，文本迅速分叉。

所以：

> token 不同不等于模型精度下降；token 完全一致也不代表中间计算没有误差。

## 3. 任务层：最终看模型质量是否回归

真正决定是否可接受，通常还要跑标准数据集，例如按模型用途选择：

- 通用能力：MMLU、CEval 等；
- 数学推理：GSM8K、MATH 等；
- 代码能力：HumanEval 等；
- 固定语料的 loss/perplexity；
- 实际业务数据集上的准确率或人工评测。

应固定 chat template、解码参数、数据集版本和评测脚本，比较新 backend 与 eager 基线。结果还要考虑评测噪声，最好使用足够大的样本、多个种子或置信区间，不能因一两个样本变化就判定回归。

**通常的验收标准是：数值误差处于合理浮点范围，没有系统性异常，并且下游任务指标相对 eager 没有超出噪声范围的显著下降。**

## 对当前 Qwen3-0.6B 情况的评价

目前证据更支持“**正常的浮点实现差异**”，而不是“Triton RMSNorm 实现错误”：

- 两边数学公式一致；
- 差异来自 FP32 归约顺序和 `rsqrt` 实现；
- 微基准中 851,968 个 BF16 输出仅有 2 个不同；
- 未发现系统性偏差；
- token 分歧发生在近似平票处；
- 保留原生 RMSNorm 后恢复 12/12 一致，证明它是本次分歧来源。

但这些结果**还不足以正式判定整模型精度可接受**，因为尚未完成标准数据集评测。比较合理的结论是：

> 当前差异符合换 backend 时常见的浮点噪声特征，没有发现算法错误；是否满足模型质量要求，还需要通过固定数据集上的任务指标回归测试确认。

因此，不建议仅为了 12 条提示词的 token 完全一致就直接取消分解。更合理的是先完成质量评测和性能对比：若质量无回归，通常保留优化路径；若确实出现显著任务指标下降，再将保留原生 RMSNorm 作为针对性规避措施。

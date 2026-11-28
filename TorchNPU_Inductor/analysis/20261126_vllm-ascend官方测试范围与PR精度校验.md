# vllm-ascend 官方 E2E/Nightly 测试范围与 PR 精度校验分析

日期：2026-11-26。

分析对象：[`vllm-project/vllm-ascend` 官方仓库](https://github.com/vllm-project/vllm-ascend)，固定提交 [`d86f630b2292ded348ffcfe20050ea338698f384`](https://github.com/vllm-project/vllm-ascend/commit/d86f630b2292ded348ffcfe20050ea338698f384)。文内源码链接均固定到该提交，避免 main 后续变化导致结论无法复核。真实 PR 运行证据使用 [#22128](https://github.com/vllm-project/vllm-ascend/pull/22128) 的最后一次 E2E 运行。

本文不包含本地 fork 的定制测试。文件数量是上述提交的静态计数，不是 pytest 参数化展开后的用例数。

## 1. 结论速览

1. **PR 门禁不默认运行全部 E2E，也不运行 Nightly。** `ready-precise` 只运行覆盖率/AST 推荐的受影响测试；`main2main` 才扫描完整 PR 候选目录。
2. **官方 PR 候选集为 219 个 `test_*.py` 文件**：单卡 94、双卡 92、四卡 33。`eight_card/` 当前没有测试文件，也没有对应 PR 路由。
3. **官方 `tests/e2e/nightly/` 有 160 个测试文件，但当前矩阵只直接引用 `singlecard_ops/` 的 149 个文件**，分别在 A3-560T、A5 上运行。
4. **PR 的“精度”主要是回归一致性检查**：固定文本/Token、Eager 与特性路径的 logprob、投机解码接受率、算子数值一致性；不是 GSM8K/MMMU 等绝对能力评测。
5. **精度用例不会在每个 PR 中全部执行。** 即使 `main2main`，仍会应用全局跳过、文件内部 `skip/xfail` 和 A5 独立选择规则。
6. **真实 #22128 PR 使用 `ready-precise` 启动了 21 个 NPU 执行组**，覆盖 A2、A3-560T、310P；不能据此推断所有模型精度测试都运行了。
7. **性能回归主要由 Nightly 承担。** PR 中明确的 Qwen3-8B 吞吐用例当前在全局跳过列表中。

## 2. 官方测试目录与候选范围

### 2.1 `tests/e2e/` 目录分工

| 目录 | 内容 | 是否进入普通 PR 全量扫描 |
|---|---|---|
| `pull_request/` | PR 功能、数值回归测试 | 是 |
| `nightly/` | Nightly 算子测试及辅助脚本 | 否 |
| `models/` | `lm_eval`、ASR、Reward Model 精度测试和 YAML | 否 |
| `common/` | 单节点、多节点公共执行器 | 不是独立测试集 |
| `doctests/` | 安装和 Quick Start 示例测试 | 否，走 Doc Test 工作流 |
| `vllm_interface/` | 上游 vLLM 接口兼容检查 | 否，走上游测试工作流 |
| `prompts/` | 测试提示词数据 | 否 |

官方选择器的全量扫描根目录只有：

```python
_ALL_TESTS_ROOTS = ("tests/ut", "tests/e2e/pull_request")
```

来源：[`select_tests.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/scripts/select_tests.py)。

### 2.2 PR 候选集

| 硬件目录 | 文件数 | 主要内容 |
|---|---:|---|
| `one_card/` | 94 | 基础推理、MRV2、Eager/Graph、Attention/MLA/SFA、量化、LoRA、Pooling、RLHF、CPU Offload |
| `two_card/` | 92 | TP2、DP/PP/EP/EPLB、投机解码、PD/EPD、Shared Expert DP、LoRA、RLHF、独立 `ops/` 和 `ops/triton/` |
| `four_card/` | 33 | TP4、DP+TP、PP+MTP、PCP/DCP、KVPP、Batch Invariant、DeepSeek-V4、Kimi-K3、MiniMax-M3 等 |
| `eight_card/` | 0 | 当前没有 `test_*.py` |

目录来源：[`tests/e2e/pull_request/`](https://github.com/vllm-project/vllm-ascend/tree/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request)。

主要模型族包括：

- **Qwen**：Qwen3、Qwen3.5、Qwen3.6、Qwen3.8、Qwen3-Next，以及 VL、Audio、Embedding、Reranker。
- **DeepSeek**：V2、V2-Lite、V3/V3.2、V4、MTP、DSpark。
- **其他**：Kimi-K3、GLM5.3、MiniMax-M3、Gemma4、GPT-OSS、Llama、Step、MiniCPM、OLMoE。

### 2.3 Nightly 候选集与实际调度

| 路径 | 文件数 | 当前矩阵是否直接引用 |
|---|---:|---|
| `single_node/ops/singlecard_ops/` | 149 | 是，A3-560T 和 A5 各运行一遍 |
| `single_node/ops/multicard_ops_a3/` | 6 | 否 |
| `310p/single_node/ops/singlecard_ops/` | 3 | 否 |
| `single_node/ops/test_mega_moe.py` | 1 | 否 |
| `single_node/ops/test_minimax_qknorm_rope_cache_performance.py` | 1 | 否 |

来源：[`tests/e2e/nightly/`](https://github.com/vllm-project/vllm-ascend/tree/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/nightly) 和 [`nightly_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/configs/nightly_config.yaml)。

这说明：**文件存在于仓库，不等于 CI 会调度；必须同时检查矩阵配置。**

## 3. PR 门禁实际如何选择测试

### 3.1 基本检查

相关 PR 会执行：

- PR 标题检查；
- Gitleaks、pre-commit、格式和静态检查；
- mypy；
- 非纯文档变更通常运行完整 `tests/ut`；
- CODEOWNERS 检查；
- 最终由 `ci-gate` 汇总结果。

main/dev 上 CPU UT 分别使用经验证的 vLLM main commit 和对应 release tag；release 分支只使用对应 release tag。

来源：[`pr_test.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/pr_test.yaml)。

### 3.2 NPU E2E 的选择模式

| 条件 | 选择范围 | 版本矩阵 |
|---|---|---|
| `ready-precise` | 根据改动行、历史覆盖率和 AST 推荐测试 | 默认使用主 vLLM ref |
| `main2main` | 扫描完整 PR 候选目录 | vLLM main 和 release 各运行 |
| 修改测试选择/执行 CI 流程 | 即使带 `ready-precise`，也升级为全量选择 | 按当前分支规则 |
| `ready-a5` | 额外运行 8 个 A5 精选文件 | 需要与上述运行标签配合 |
| `/e2e <路径>` | 只运行手动指定文件或用例 | 用于调试，不是默认门禁 |

核心调用为：

```bash
# ready-precise
select_tests.py --test-list-file <推荐列表> --skip-cpu-ut --skip-a5

# main2main 或 CI 流程变更
select_tests.py --all-tests --skip-cpu-ut --skip-a5

# ready-a5
select_tests.py --curated a5 --runner-label-override linux-aarch64-a5-4
```

`accuracy_tests` 是精度关键文件记录和路由兼容配置，**不是“强制全部执行”列表**。

### 3.3 机器映射

| 测试路径 | 机器 |
|---|---|
| 普通 `one_card/` | ARM64 A2 单卡 |
| 普通 `two_card/` | ARM64 A3-560T 双卡 |
| 普通 `four_card/` | ARM64 A3-560T 四卡 |
| `_310p` 单/双/四卡 | ARM64 310P 对应卡数 |
| 特定 HiSparse、FlashAttention、Expert Offload 等 | A5 |
| CPU UT | AMD64 CPU runner |

路由来源：[`test_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/scripts/test_config.yaml)。

### 3.4 全量模式仍然存在的例外

全局跳过文件：

```text
four_card/model_runner_v2/test_kvpp_pp.py
four_card/test_graph_mode.py
one_card/_310p/test_classification_310p.py
one_card/model_runner_v2/test_uva.py
two_card/test_qwen3_performance.py
two_card/lora/test_llama32_lora_tp2.py
two_card/aclgraph/test_aclgraph_capture_replay.py
four_card/spec_decode/test_mtp_step3p5.py
```

文件内部还会跳过 Qwen3-8B EAGLE3 SP acceptance、Qwen3.6 DSpark adaptive GDN logprob 等用例。普通 PR 全量扫描也不包含 `nightly/`、`models/` 的 Nightly 精度组、`doctests/` 和 `vllm_interface/`。

## 4. PR 精度校验：模型、基线、控制变量与容差

官方 `accuracy_tests` 标记了 24 个文件，其中 3 个当前全局跳过。需要先区分四类基线，不能把它们都理解为同一种 A/B 实验。

### 4.1 实时 A/B：Eager 基线与待验证路径

公共函数 [`compare_logprobs()`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/utils.py) 的控制方式如下：

| 项目 | 基线 A | 待验证 B |
|---|---|---|
| 模型和权重 | 同一模型路径、同一份权重 | 与 A 相同 |
| 执行方式 | 强制 `enforce_eager=True` | 使用传入的 Graph/编译配置 |
| 特性 | 删除 `additional_config` | 保留完整配置 |
| 并行配置 | 保留原测试 TP 等参数 | 与 A 相同 |
| 输入 | 同一组 prompts | 与 A 相同 |
| 采样 | `temperature=0`，3 Token，Top-20 logprob | 与 A 相同 |
| 环境 | 同一 CI 容器/runner，依次启动两个引擎 | 与 A 相同 |

判据：

- Prefill 第 1 个 Token：Token ID 必须完全相同，logprob 绝对误差默认 **≤0.0689**。
- Decode 后 2 个 Token：Token 相同时直接比较 logprob；Token 不同时，对同一基线 Token 做交叉比较。默认绝对误差 **≤0.1378**。

| 模型与场景 | A | B | 容差 |
|---|---|---|---|
| DeepSeek-V2-Lite，Shared Expert DP | TP2+EP+Eager，不开 Shared Expert DP | 分别启用 Eager+Shared Expert DP、Graph+Shared Expert DP | Prefill `0.0689`，Decode `0.1378` |
| Qwen3.8-27B，FlashComm SP | TP2+Eager，不开 FlashComm | TP2+Eager/Graph+FlashComm | 同上 |
| Qwen3-0.6B，XLite | 默认 Eager，移除 `additional_config` | XLite Decode-only + `weight_nz_mode=2` | 同上；XLite 与 NZ 布局同时变化 |
| Qwen3-0.6B，CPU Weight Offload | 相同 NZ 模式、无 Offload 的 Eager | Prefetch Offload 的 Eager/Graph | 同上；按 NZ=0/2 分别建立基线 |

来源：[`test_shared_expert_dp.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/two_card/test_shared_expert_dp.py)、[`test_qwen3_8_27b_dense_sp.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/two_card/test_qwen3_8_27b_dense_sp.py)、[`test_cpu_weight_offload.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/one_card/test_cpu_weight_offload.py)。

**能证明：** 特定执行路径没有改变短序列输出分布。

**不能证明：** 长序列生成质量、数据集准确率或性能。

### 4.2 历史 Golden：当前代码与过去接受的输出比较

这类用例不会在当前 PR 中重新运行 Eager 基线，而是比较当前输出与测试文件中的 Token ID、文本或接受率 Golden。

| 模型与场景 | 校验内容 | 容差 |
|---|---|---|
| Qwen3-0.6B，MRV2 Graph | 4 个提示的生成文本前 10 字符 | 0，`misses == 0` |
| DeepSeek-V2-Lite-Chat，PP2 | 生成 Token ID 和完整文本匹配 Golden | 0 |
| DeepSeek-V2-Lite-Chat，DP2+PP2 | 使用独立 Golden，完全匹配 | 0 |
| DeepSeek-V4-Flash-W4A8-MTP，TP4 | 两个提示的 greedy Token ID | 0 |
| DeepSeek-V4 DSA-PCP + MTP/DSpark | 输出以指定前缀开头 | 前缀完全匹配 |
| DeepSeek-V3.2-W8A8-Pruning，SFA+PCP+DCP+MTP | 完全匹配 4 组参考输出之一 | 0 |
| 随机权重 DeepSeek MTP + PCP | 前 5 Token 解码结果匹配 Golden | 0；只验证一致性，不验证语义质量 |
| Qwen3-8B + EAGLE3 + PCP | 前 5 Token 匹配两组参考之一，同时检查接受率 | 文本完全匹配，接受率下降 `<0.03` |
| Qwen3-8B + PCP，PD 分离 | 输出等于 `" Alex, and I am"` | 0 |

LoRA 的判据类似：

| 模型 | 校验内容 | 容差 |
|---|---|---|
| iLlama-3.2-1B + LoRA，TP2 | 3 条 SQL 与固定字符串完全相同 | 0 |
| Qwen3.5-4B + LoRA，TP1/TP2 | 3 条 SQL 完整输出完全相同 | 0 |
| Qwen3-30B-A3B + LoRA，TP2/EP | SQL 输出以参考字符串开头 | 前缀完全匹配 |
| Qwen3-30B-A3B，多 LoRA ID 混批 | LoRA 请求匹配各自 SQL；无 LoRA 请求不能误走 LoRA，且两者输出不同 | 精确前缀/不等式检查 |

固定 Golden 主要固定模型、提示、采样和特性配置；历史输出产生时的 CANN、驱动、硬件环境并未在当前测试中完整重建。因此它属于**回归检查**，不是严格同期 A/B。

### 4.3 投机解码接受长度

接受长度定义为：

\[
L = 1 + \frac{\text{接受的 draft Token 数}}{\text{draft 轮数}}
\]

公共检查：

\[
\frac{|L_{actual}-L_{golden}|}{L_{golden}} \leq RTOL
\]

默认 `RTOL=8%`，部分模型单独覆盖。

| 模型与场景 | 基线 | 容差 |
|---|---:|---:|
| Qwen3-8B + EAGLE3，MRV2 | `2.03` | 相对误差 **≤8%**，且实际值 `>1.0` |
| Qwen3.6-35B-A3B + DFlash，TP2 | `4.0` | **≤8%** |
| Qwen3.6-27B + DSpark，TP2 | `3.47` | **≤8%** |
| Qwen3.6-35B-A3B + DSpark，TP2 | `3.93` | **≤8%** |
| DeepSeek-V4-Flash-DSpark-W4A8，TP4 | `3.33` | **≤8%** |
| Qwen3.6-35B-A3B + MTP | `3.14127177` | **≤10%** |
| Qwen3.6-27B + MTP | `3.16079020` | **≤10%** |

RecoverSSM/ReplaySSM 参数化用例中，**接受率基线只在 `use_replayssm=False` 时检查**；开启时主要检查生成有效及 draft 活动确实发生。

来源：[`test_qwen3_6_dflash.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_dflash.py)、[`test_qwen3_6_mtp.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/two_card/model_runner_v2/test_qwen3_6_mtp.py)、[`test_spec_decode_heavy.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/two_card/spec_decode/test_spec_decode_heavy.py)。

### 4.4 投机解码逐位置接受率

位置接受率：

\[
r_i=\frac{\text{位置 }i\text{ 接受的 Token 数}}{\text{draft 轮数}}
\]

#### 双向绝对误差

| 模型与场景 | 逐位置基线 | 容差 |
|---|---|---|
| Qwen3-30B-A3B + VWN EAGLE3 | `[0.75, 0.50, 0.30]` | 每个位置 `abs(actual-golden) < 0.06` |
| Qwen3.8-27B + DFlash2 | `[0.855, 0.593, 0.359, 0.290, 0.193, 0.152, 0.076]` | 每个位置 `abs(actual-golden) < 0.20` |

这里比较的是接受率绝对差值，不是相对百分比。

#### 只允许有限下降

判据为：

```python
actual >= golden or golden - actual < tolerance
```

| 模型与场景 | 逐位置基线 | 允许下降量 |
|---|---|---:|
| Qwen3-Next-80B-A3B + MTP | `[0.85, 0.46, 0.19]` | `<0.06` |
| DeepSeek-V4-Flash-W4A8 + MTP | `[0.85, 0.65, 0.35]` | `<0.03` |
| DeepSeek-V4-Flash-DSpark-W4A8 | `[0.73, 0.64, 0.55, 0.49, 0.42]` | `<0.03` |
| Qwen3-8B + EAGLE3 + PCP | `[0.54, 0.29, 0.15]` | `<0.03` |

接受率验证的是投机机制是否退化，**不能单独证明文本正确**；部分用例因此同时检查固定文本前缀。

另一个特殊用例是 Qwen3-8B-W8A8 + EAGLE3：参考路径为非投机 `FULL_DECODE_ONLY`，待验证路径为投机 `FULL`，4 个提示中至少 3 个 Token 前缀完全匹配。这里同时改变了投机解码和图模式。

### 4.5 算子级数值一致性

| 测试 | 参考 | 容差 |
|---|---|---|
| Qwen3-8B Attention V1 | 同一 Q/K/V、KV Cache、batch metadata，NPU Attention vs PyTorch SDPA | `rtol=0.01, atol=0.01` |
| MLA Prolog Head Padding | 原生融合算子 vs 独立 CPU 投影、归一化和独立 RoPE | NOPE：`rtol=0.02, atol=0.01`；ROPE：`rtol=0.02, atol=0.02` |
| DCP Q Replication | 相同配置下关闭/开启 `dcp_q_replicate`，以关闭时输出作为实时 Golden | 文本完全匹配 |

其他 MLA/SFA/GDN/量化/Triton 用例按文件独立设置容差，没有统一阈值。

### 4.6 没有固定精度阈值或当前跳过

- **Qwen3.8-27B + LiLiCorr**：接受长度只通过 `record_property` 记录，没有固定接受率基线。greedy 首个差异 Token 允许近似并列，要求候选都在 Top-5，logprob gap 在 `[-1e-5, 0.25]`。
- **Qwen3-8B + DSpark Batch-sharded Sampling**：只检查组合功能可运行，没有文本、logprob 或接受率断言。
- **DeepSeek-V4 DSpark Adaptive Verification**：adaptive 分支只检查运行完成，非 adaptive 分支才检查接受率。
- **全局跳过**：`test_graph_mode.py`、`test_llama32_lora_tp2.py`、`test_mtp_step3p5.py`。
- **文件内部跳过**：Qwen3-8B EAGLE3 SP acceptance、Qwen3.6 DSpark adaptive GDN logprob。

## 5. Nightly 的精度与性能标准

### 5.1 Nightly 矩阵

| 平台 | 调度内容 |
|---|---|
| A2 | 9 个单节点模型/性能组、2 个双节点组、12 个模型的 `lm_eval` 精度测试 |
| A3 | 13 个多节点/双节点模型组 |
| A3-560T | 21 个单节点组（含 149 文件单卡算子集）、12 个多卡组 |
| 310P | 7 个模型组 |
| A5 | 16 个单节点组（含同一套 149 文件算子集）；多节点配置当前被注释 |

这些 Nightly 工作流由外部自动化通过 `workflow_dispatch` 触发。`/nightly all` 派发 A2、A3、310P、A5，不包含 A3-560T；后者需要 `--a3-560t`。来源：[`pr_nightly_command.yml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/pr_nightly_command.yml)。

### 5.2 A2 默认 `lm_eval` 模型精度

| 数据集 | 模型与基线 |
|---|---|
| GSM8K | Qwen3-8B：0.765/0.810；Qwen3-8B-W8A8：0.800/0.820；Qwen2-Audio-7B：0.440/0.450；Minitron-8B：0.5436/0.5451；Qwen3-30B-A3B：0.890/0.850；Qwen3-30B-A3B-W8A8：0.900/0.800。斜杠两侧为 strict/flexible exact match |
| MMMU | Qwen3-VL-8B-W8A8：0.52；Qwen3-VL-8B：0.55；Qwen3-VL-30B-A3B：0.58；Qwen3-Omni-30B：0.60 |
| C-Eval | Qwen3-Next-80B accountant：0.98；Mixtral-8x7B：0.45；Qwen3-30B-A3B 另检查 0.84 |

`lm_eval` 使用：

```python
task_success = np.isclose(ground_truth, measured_value, rtol=0.05)
```

来源：[`test_lm_eval_correctness.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/models/test_lm_eval_correctness.py)。

另有 7 个 PR-only 精度模型：Gemma-3-4B、InternLM3-8B、Qwen3-ASR-1.7B、InternVL3.5-8B、LLaVA-OneVision、Qwen2.5-Math-RM-72B、Hunyuan-A13B。它们不属于 `/nightly all`，也不属于普通 PR 自动精度全集，需要显式指定名称。

### 5.3 AISBench 精度与性能

来源：[`tools/aisbench.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tools/aisbench.py)。

- 默认精度：`baseline - threshold ≤ accuracy ≤ baseline + threshold`。这里是绝对差值。
- `verify_mode: ge`：只要求 `accuracy ≥ baseline`。
- 输出吞吐：`Output Token Throughput ≥ baseline × threshold`。
- 输入吞吐：可选 `Input Token Throughput ≥ input_baseline × input_threshold`。
- TPOT：可选 `TPOT ≤ tpot_threshold`，单位 ms。
- TTFT：YAML 还可比较两个 benchmark 的 TTFT，支持 `<`、`<=`、`>`、`>=` 和比例系数。

示例：Qwen3-30B-A3B BF16 A2 的输出吞吐基线为 `765 token/s`、阈值 `0.97`，最低通过值为 `742.05 token/s`。来源：[`Qwen3-30B-A3B-BF16-A2.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/SIT/cases/models/configs/Qwen/Qwen3-30B-A3B-BF16-A2.yaml)。

PR 中的 Qwen3-8B 吞吐测试基线为 `1514 token/s`、默认阈值 `0.97`，但该文件当前在全局跳过列表中，不能视为现行 PR 性能门禁。

### 5.4 Daily 不是 Nightly 全量

[`daily_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/configs/daily_config.yaml) 只列了 A3-560T 的少量高频用例，并且 310P 不由 Daily 工作流派发。每日任务还会运行 PR 完整测试集，因此“Daily”与“Nightly 全量”不能混用。

## 6. 真实 PR 运行证据：#22128

PR：[Support W4A8 MegaMoe shared-expert fusion](https://github.com/vllm-project/vllm-ascend/pull/22128)，最终提交 `3628fac`，带 `ready-precise`。

实际运行：[E2E run 45357917678](https://github.com/vllm-project/vllm-ascend/actions/runs/45357917678)。

| 机器类型 | 执行组 | 实际 runner 示例 | 设备测试耗时示例 |
|---|---:|---|---:|
| A2 单卡 | 7 组 | `linux-aarch64-a2b1b3-fed-1-runner-wcmfl` | 5 分 18 秒 |
| A3-560T 双卡 | 5 组 | `linux-aarch64-a3-800i-fed-2-runner-wcgtp` | 30 分 46 秒 |
| A3-560T 四卡 | 5 组 | `linux-aarch64-a3-800i-fed-4-runner-m9gg9` | 45 分 39 秒 |
| 310P 单卡 | 2 组 | `linux-aarch64-310p-1-n6rm5-runner-9jnbt` | 18 分 35 秒 |
| 310P 双卡 | 1 组 | `linux-aarch64-310p-2-j5g5b-runner-7kjxz` | — |
| 310P 四卡 | 1 组 | `linux-aarch64-310p-4-vrbgc-runner-mhcd7` | 9 分钟 |

此外：

- CPU UT 对 vLLM main `c410543...` 和 `v0.33.0` 各运行一次，机器池为 `linux-amd64-cpu-8-hk`。
- 上游 vLLM E2E 在 `linux-aarch64-a2b3-fed-1-runner-sc6js` 上运行。
- A5 任务跳过，因为没有 `ready-a5`。
- 21 个 NPU 执行组均成功，但不能据此推断全部 219 个 PR 文件或全部精度文件都执行了。

每个分区的精确 pytest 文件清单在 `selected-test-logs-*` 附件中；匿名下载附件会返回 401，需要 GitHub 登录后核查。

## 7. 最终判读规则

| 问题 | 正确判读 |
|---|---|
| 文件在 `tests/e2e/nightly/`，CI 会运行吗？ | 不一定，检查 `nightly_config.yaml` 或 weekly/daily 配置 |
| 文件在 `pull_request/`，每个 PR 会运行吗？ | 不一定，检查 `ready-precise` 推荐结果或是否使用 `main2main` |
| 在 `accuracy_tests` 中，一定会运行吗？ | 不一定，该列表不是强制执行开关 |
| logprob 对比能证明数据集精度吗？ | 不能，只证明特定短序列的输出分布一致性 |
| 接受率达标能证明文本正确吗？ | 不能，只证明投机机制效率未显著退化 |
| PR 会运行 GSM8K/MMMU 吗？ | 普通 PR 门禁不运行这些 Nightly 模型组 |
| 全量 PR 等于全部 219 文件吗？ | 不是，仍有全局跳过、文件内部跳过和 A5 独立规则 |

## 8. 主要代码与运行来源

- [官方测试贡献指南](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/docs/source/developer_guide/contribution/testing.md)
- [PR 工作流 `pr_test.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/pr_test.yaml)
- [测试选择器 `select_tests.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/scripts/select_tests.py)
- [PR 路由与跳过配置 `test_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/scripts/test_config.yaml)
- [Nightly 矩阵 `nightly_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/.github/workflows/configs/nightly_config.yaml)
- [PR logprob 公共实现 `pull_request/utils.py`](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/pull_request/utils.py)
- [`lm_eval` 精度实现](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/models/test_lm_eval_correctness.py)
- [AISBench 精度与性能实现](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tools/aisbench.py)
- [单节点功能与 benchmark 执行器](https://github.com/vllm-project/vllm-ascend/blob/d86f630b2292ded348ffcfe20050ea338698f384/tests/e2e/common/single_node/test_single_node.py)
- [#22128 实际 E2E 运行](https://github.com/vllm-project/vllm-ascend/actions/runs/45357917678)

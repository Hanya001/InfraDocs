# benchmark：Inductor 测试工具

本目录提供 4 个单模型运行脚本和 7 个独立 Python 工具。测试范围见《[Inductor适配测试覆盖模型与量化矩阵.md](../Inductor适配测试覆盖模型与量化矩阵.md)》。

- `run_correctness.sh`：eager 与 Inductor 固定输入正确性。
- `run_accuracy.sh`：eager 与 Inductor 的 MMLU、GSM8K 任务分数。
- `run_performance.sh`：default 与 Inductor 的 ShareGPT、长上下文性能。
- `run_offline_performance.sh`：default 与 Inductor 的固定 batch Latency、ShareGPT 离线 Throughput。

Python 工具之间没有代码导入；脚本通过命令行、JSON 文件和缓存目录组合。测试结果不要写入本目录。

## 1. 快速开始

### 准备环境

脚本在 A5 容器内运行。先同步本目录到服务器，并进入容器：

```bash
scp -r . <host>:<新的服务器脚本目录>
ssh -t <host> 'docker exec -it <container> bash'
```

容器内需要有：

- 模型权重和 workspace 中的 `env.sh`。
- AISBench、MMLU、GSM8K 和 ShareGPT 数据。
- `zstandard`，以及 `xgrammar==0.2.8`。

MMLU 目录应包含 `dev/`、`test/`；GSM8K 目录应包含 `train.jsonl`、`test.jsonl`。

仓库还保存了两套固定样本，供重复测试直接复用：

- `fixed200/`：MMLU、GSM8K 各 200 题，来自 20261122 的 Qwen3.8-27B-W8A8 对比实验；
- `fixed1000_500/`：MMLU 1000 题、GSM8K 500 题，包含 `fixed200/` 中的原有样本。

每套目录都包含 MMLU/GSM8K 清单、实际题目及对应 dev/train 数据。相同样本量的复跑应优先使用固定数据，不重复抽样。

`run_accuracy.sh` 和 `accuracy.py run` 支持 `--fixed-data-root` 直接复用上述目录。使用 `fixed1000_500` 时，同时指定 `--mmlu-num-prompts 1000 --gsm8k-num-prompts 500`；两个覆盖参数也可单独用于非固定抽样。

默认情况下，任何非零的 AISBench response anomaly 指标都会使任务失败。需要保留异常输出并继续后续任务时，显式指定 `--allow-response-anomalies`；异常指标会写入 `accuracy_results.json` 的 `response_anomalies` 字段，不会被忽略。

指定模型增加 GPQA Diamond 时，使用 `--tasks mmlu,gsm8k,gpqa --gpqa-dataset <包含gpqa_diamond.csv的目录> --gpqa-num-prompts 198`。GPQA 固定运行完整 Diamond 集，使用 AISBench 官方 `gpqa_gen_0_shot_cot_chat_prompt` 配置，推理输出上限为 32768 tokens；MMLU/GSM8K 的 2048 上限不适用于 GPQA。

### 运行一个模型

下面以 Qwen3.8-27B-W8A8、TP2、NPU 0/1 为例。各结果目录都必须是新目录：

```bash
cd <workspace>
source env.sh
WORKSPACE=$PWD
SCRIPTS=<benchmark脚本绝对路径>
MODEL=/mnt/weight/Qwen3.8-27B-w8a8
BATCH=<新的批次绝对路径>
AISBENCH_HOME=<AISBench目录>
MMLU=<MMLU目录>
GSM8K=<GSM8K目录>
SHAREGPT=<ShareGPT.json绝对路径>

python "$SCRIPTS/runtime.py" check-devices --devices 0,1

bash "$SCRIPTS/run_correctness.sh" \
  --model "$MODEL" --workspace "$WORKSPACE" \
  --results "$BATCH/correctness" --devices 0,1 --tp 2

bash "$SCRIPTS/run_accuracy.sh" \
  --model "$MODEL" --workspace "$WORKSPACE" \
  --results "$BATCH/accuracy" \
  --aisbench-home "$AISBENCH_HOME" --mmlu "$MMLU" --gsm8k "$GSM8K" \
  --devices 0,1 --tp 2 --cache-root "$BATCH/correctness/cache"

bash "$SCRIPTS/run_performance.sh" \
  --model "$MODEL" --workspace "$WORKSPACE" \
  --results "$BATCH/performance" --sharegpt "$SHAREGPT" \
  --devices 0,1 --tp 2 --cache-root "$BATCH/correctness/cache"

bash "$SCRIPTS/run_offline_performance.sh" \
  --model "$MODEL" --workspace "$WORKSPACE" \
  --results "$BATCH/offline_performance" --sharegpt "$SHAREGPT" \
  --devices 0,1 --tp 2 --cache-root "$BATCH/correctness/cache"
```

首次验证环境时，可把任务集命令改为 `--tasks mmlu --num-prompts 4`，并使用新的结果目录。

## 2. 四个脚本做什么

| 脚本 | 默认流程 | 主要结果 |
|---|---|---|
| 正确性 | PR 的 4 条短 prompt 加原有 4 条长 prompt，共 8 组，eager → Inductor，各生成 2 轮，再比较 | token、文本、logprob、首个分叉点、实际引擎配置 |
| 任务集 | 固定 MMLU/GSM8K 各 200 题，eager → Inductor | 总分、百分点差值、固定样本清单 |
| 性能 | ShareGPT QPS 1/4/16/inf 和两组长输入，执行三轮 ABAB | 官方性能 JSON、逐轮比较、中位数和波动、Summary |
| 离线性能 | Latency 32/128、batch 8、预热 5 轮、测量 15 轮；Throughput 使用 ShareGPT 200 条、0 条预热；执行三轮 ABAB | 官方 JSON、逐轮比较、中位数和波动、Summary |

顺序为：

```text
default → Inductor → default → Inductor
```

MMLU 从全部学科固定抽样，再合并为一个 `fixed200` 数据集，一次推理、一次评分。它使用通用选择题提示词，只用于同配置 eager/Inductor 比较，不作为公开 MMLU 分数。

Embedding 正确性使用 `run_correctness.sh --embedding`。

所有脚本都支持 `--help`。常用参数：

- `--devices`：物理卡列表，数量必须等于 `--tp`。
- `--cache-root`：复用已有缓存；省略则使用本次结果目录下的新缓存。
- `--enable-prefix-caching`：开启 prefix cache，默认关闭。
- `--enable-expert-parallel`、`--quantization`、`--trust-remote-code`、`--kv-cache-dtype`、`--tokenizer-mode`：传递对应的 vLLM 模型运行参数；W8A8C8/W4A8C8 使用 `--kv-cache-dtype int8`，DeepSeek-V4.1 官方 MXFP 权重使用 `--quantization deepseek_v4_fp8 --tokenizer-mode deepseek_v41`。
- `--max-model-len`、`--gpu-memory-utilization`：按需覆盖上下文长度和 NPU 显存利用率；default 与 Inductor 必须保持一致。
- `--dp N`：设置数据并行度，设备数必须等于 `TP × DP`，用于 DeepSeek-V4.1 官方推荐的 DP8/TP1/EP8 等拓扑。
- `--repeats`：正确性重复生成次数或性能轮数；两个性能脚本默认 3 轮，也允许显式指定 1 轮。
- 在线性能脚本每次停止服务后，最多等待 300 秒让所用 NPU 的进程退出且 HBM 回到空闲水平，再启动下一条路径或下一轮，避免显存尚未回收导致后续服务 OOM。
- `--tests LIST`：离线性能脚本专用，可选 `latency,throughput` 中的一个或两个。
- `--load-format NAME`：离线性能脚本专用，可设置 `dummy`；省略时使用真实模型权重。
- `--capture-size N`：性能脚本专用。同时给 default 和 Inductor 设置图捕获上限，并展开为尺寸阶梯 `[1,2,4,8,16,24,32]` 加不超过 N 的 2 的幂（N 为 32 或 ≥64 的 2 的幂）。省略时两条路径都保持 vLLM 默认值（32）。比较 Inductor 的净收益时应固定该值，否则捕获窗口不同会混淆结论。
- `--qps LIST`：性能脚本专用，逗号分隔的 ShareGPT 到达率，默认 `1,4,16,inf`；填 `none` 跳过 ShareGPT。
- `--skip-long`：性能脚本专用，跳过 random 8192/32768 长上下文负载。
- `--timeout`：进程超时秒数。

## 3. 看结果和退出码

每个阶段都会生成 `SUMMARY.md`，原始结果保留在同目录下。

- 正确性重点看 token 一致数、logprob 差异和分叉点候选。采样固定 `temperature=0`、`top_p=1.0`、`top_k=0`、`seed=0`、`max_tokens=1024`、`ignore_eos=False`，并返回 Top-20 logprobs；`logprob_within_tolerance` 按 PR 规则判定：prefill Token ID 必须相同且 logprob 绝对误差不超过 0.0689；decode 误差不超过 0.1378，Token 不同时 eager Token 必须出现在 Inductor 的 Top-20 中。每个 prompt 的 `pr_failure` 记录首个失败原因，原有长文本和分叉统计保留。
- 任务集重点看两条路径的分数差；`selection_sha256` 相同表示样本集合一致。
- 性能统一报告 Inductor 相对 default 的加速比，三项均为越高越好：吞吐加速比=`Inductor/default`，TPOT、TTFT 加速比=`default/Inductor`。大于 1 表示 Inductor 更快。
- 默认阈值分别为吞吐加速比 >=0.95、TPOT 加速比 >=1/1.05（约 0.95238）、TTFT 加速比 >=1/1.10（约 0.90909），与原“TPOT 劣化不超过 5%、TTFT 劣化不超过 10%”等价。
- 报告中加速比 >1.05 使用粗体，<0.95 使用斜体；等于边界时不标记。
- `inductor_evidence.json` 和 `server.json` 用于确认实际 backend、图模式和生成产物。

**数值不一致默认只写报告，脚本仍正常退出。** 参数、服务、请求或结果文件错误才返回非零。

逐样本统计取决于 AISBench 输出：没有逐题 `details` 时，`sample_comparison.available=false`，需要另外分析 `predictions/`。

## 4. 缓存和归档

缓存按编译路径分开：

```text
cache/eager/
cache/default/
cache/inductor/
```

推荐先运行正确性，再把它的 `cache/` 传给任务集和性能。每个 `runtime.json` 记录实际缓存路径及 cold/warm 状态。

将 JSON、JSONL、日志、环境快照和 Summary 归档到 `server-results/<主题>/<日期_批次>/`。大体积缓存通常留在服务器，不必同步到本地。

## 5. 单独使用 Python 工具

| 工具 | 用途 |
|---|---|
| `correctness.py` | `generate`、`embedding`、`compare` |
| `accuracy.py` | `run`、`rescore`、`compare` |
| `performance.py` | `run`、`compare` |
| `offline_performance.py` | `run`、`compare`、`summary` |
| `server.py` | `run`、`wait` |
| `runtime.py` | 设备检查、命令包装、环境快照、证据收集 |
| `report.py` | 汇总显式指定的 JSON |

例如，只比较两份已有正确性结果：

```bash
python correctness.py compare --type generation \
  --reference <eager.jsonl> --candidate <inductor.jsonl> \
  --out <compare.json>
```

只比较两份官方性能结果：

```bash
python performance.py compare \
  --reference <default.json> --candidate <inductor.json> \
  --out <compare.json>
```

完整参数使用 `<工具> --help` 或 `<工具> <子命令> --help` 查看。

## 6. 常见问题

### AISBench 依赖

```bash
pip install zstandard
pip install xgrammar==0.2.8
```

预测已落盘但未评分时，可离线补算：

```bash
python accuracy.py rescore \
  --run-dir <已有任务结果目录> --task mmlu --num-prompts 200 \
  --out <rescore.json>
```

### 空闲卡仍有 HBM 占用

A5 空闲卡通常约有 4.8 GB 基础占用，默认检查阈值为 8192 MB：

```bash
python runtime.py check-devices --devices 0,1 --max-used-mb 8192
```

返回非零时不要启动测试，也不要终止不属于自己的进程。

### 长上下文请求数

性能脚本默认 random 为并发 8、32 条正式请求、8 条预热。若严格使用矩阵中的 32768/1024 配置，应改为 16 条正式请求、4 条预热：

```bash
python performance.py run --model "$MODEL" --base-url "$BASE_URL" \
  --dataset-name random --random-input-len 32768 --random-output-len 1024 \
  --num-prompts 16 --num-warmups 4 --max-concurrency 8 \
  --request-rate inf --ignore-eos --out <结果.json>
```

默认参数得到的结果不能冒充该矩阵配置的结果。

## 7. 部署

本地修改后同步到服务器的新脚本目录，避免新旧文件混放。多模型调度、远程连接和并行由外层 shell 或 CI 负责，本目录不提供总控脚本。

# benchmark：Inductor 测试工具

本目录提供 6 个可独立组合的 vllm-ascend Inductor 测试工具，以及可选的 `run_performance.sh` 单模型性能基准和 `run_matrix.sh` 批量编排脚本。测试范围与验收标准见《[Inductor适配测试覆盖模型与量化矩阵.md](../Inductor适配测试覆盖模型与量化矩阵.md)》；测试结果只归档到本机 `D:\HuaweiWorkSpace\Inductor\server-results\benchmark\`，不要写入本目录。

## 使用原则

- 按功能组织脚本，通过子命令区分具体操作，避免为每个原子操作新建文件。
- 每个子命令都可独立执行，不依赖其他测试先生成隐式状态，也不会自动执行下一步。
- 模型、设备、缓存、输入和输出通过命令行明确指定；没有内置模型清单、服务器地址或固定结果根目录。
- 数据生成与比较分离，服务与客户端分离，批量调度由外层 shell 或 CI 组织。
- 默认不覆盖已有结果。比较操作只有显式传入 `--require-*` 才把数值差异转换为非零退出码。

查看完整参数：

```bash
python correctness.py --help
python correctness.py generate --help
```

实际运行 vLLM 前，在 workspace 根目录执行 `source env.sh`；也可以让 `runtime.py exec --workspace ...` 为子进程加载该环境。

## 6 个工具

| 工具 | 子命令 | 职责 |
|---|---|---|
| `correctness.py` | `generate` | 生成 8 组固定输入的文本、token 和 logprob；支持重复生成、图模式、TP、MTP、prefix cache |
|  | `embedding` | 生成固定输入的 Embedding 向量 |
|  | `compare` | 比较两份生成结果或 Embedding；通过 `--type generation\|embedding` 选择 |
| `server.py` | `run` | 以前台方式运行一个 OpenAI 兼容 vLLM 服务 |
|  | `wait` | 通过 `/v1/models` 等待服务就绪，不检查日志内容 |
| `accuracy.py` | `run` | 对已有服务执行 AISBench 任务 |
|  | `rescore` | 只对已有预测离线补算分数，不重跑推理 |
|  | `compare` | 比较两份任务分数并计算百分点差值 |
| `performance.py` | `run` | 对已有服务执行一次官方 `vllm bench serve` 负载 |
|  | `compare` | 比较官方结果中的吞吐、TPOT 和 TTFT |
| `runtime.py` | `exec` | 为任意命令提供缓存隔离、可选逐卡锁、超时和进程组清理 |
|  | `inspect` | 检查权重分片和索引，记录架构与量化字段 |
|  | `snapshot` | 记录 Python、软件包、git 版本和运行时环境 |
|  | `evidence` | 收集 backend、生成产物、capture/replay 等 Inductor 路径证据 |
| `report.py` | 无子命令 | 只汇总命令行明确列出的比较 JSON |

## 固定输入正确性

以下命令在 A5 容器内执行。`SCRIPTS` 指向本目录，`RESULTS` 使用新的结果目录。

```bash
cd <workspace>
SCRIPTS=<benchmark脚本目录>
RESULTS=<结果目录>
MODEL=/mnt/weight/Qwen/Qwen3-0.6B

python "$SCRIPTS/runtime.py" inspect \
  --model "$MODEL" --expected-name Qwen3-0.6B --out "$RESULTS/weights.json"
source env.sh
python "$SCRIPTS/runtime.py" snapshot --workspace "$PWD" --out "$RESULTS/environment.json"

python "$SCRIPTS/runtime.py" exec \
  --cache-dir "$RESULTS/cache/eager" --out "$RESULTS/eager.runtime.json" \
  --compile-debug-dir "$RESULTS/eager.compile_debug" --workspace "$PWD" -- \
  python "$SCRIPTS/correctness.py" generate \
    --model "$MODEL" --track eager --out "$RESULTS/eager" --repeats 2

python "$SCRIPTS/runtime.py" exec \
  --cache-dir "$RESULTS/cache/inductor" --out "$RESULTS/inductor.runtime.json" \
  --compile-debug-dir "$RESULTS/inductor.compile_debug" --workspace "$PWD" -- \
  python "$SCRIPTS/correctness.py" generate \
    --model "$MODEL" --track inductor --out "$RESULTS/inductor" --repeats 2

python "$SCRIPTS/correctness.py" compare --type generation \
  --reference "$RESULTS/eager/acc_track.jsonl" \
  --candidate "$RESULTS/inductor/acc_track.jsonl" \
  --reference-label eager --candidate-label inductor \
  --out "$RESULTS/generation_compare.json"
```

需要把差异直接作为 CI 失败时，增加 `--require-token-match` 或 `--require-logprob-tolerance`。不加这些参数时只保存差异，不自动阻断后续独立任务。

Embedding 使用同一个工具：

```bash
python "$SCRIPTS/correctness.py" embedding \
  --model "$MODEL" --track eager --out "$RESULTS/embedding_eager"
python "$SCRIPTS/correctness.py" embedding \
  --model "$MODEL" --track inductor --out "$RESULTS/embedding_inductor"
python "$SCRIPTS/correctness.py" compare --type embedding \
  --reference "$RESULTS/embedding_eager/embeddings.json" \
  --candidate "$RESULTS/embedding_inductor/embeddings.json" \
  --out "$RESULTS/embedding_compare.json" --require-within-tolerance
```

## 图模式、TP 与运行包装

图模式只是引擎参数，不需要单独的批处理脚本：

```bash
python "$SCRIPTS/correctness.py" generate \
  --model "$MODEL" --track inductor --cudagraph-mode PIECEWISE \
  --tp 2 --repeats 2 --out "$RESULTS/piecewise_tp2"
```

覆盖多种图模式时，由外层循环调用，并为每种编译配置指定独立缓存。MTP 使用 `--mtp --mtp-tokens N`，prefix cache 使用 `--enable-prefix-caching`。

需要同时指定设备、缓存和超时时，只使用一层 `runtime.py exec`：

```bash
python "$SCRIPTS/runtime.py" exec \
  --devices 2,3 --lock-dir <锁目录> --timeout 3600 \
  --cache-dir "$RESULTS/cache/tp2" --out "$RESULTS/tp2.runtime.json" --workspace "$PWD" -- \
  python "$SCRIPTS/correctness.py" generate \
    --model "$MODEL" --track inductor --tp 2 --out "$RESULTS/tp2"
```

## 服务、任务精度与性能

`server.py run` 保持前台运行，下面放到 shell 后台。`SERVER_PID` 就是实际 vLLM 进程；就绪判断使用 HTTP，不等待日志中的特定字符串。端口为 0 时从 `server.json` 读取实际端口。

```bash
cd <workspace>
source env.sh
BASE_URL=http://127.0.0.1:8000
python "$SCRIPTS/server.py" run \
  --model "$MODEL" --track inductor --port 8000 --no-enable-prefix-caching \
  --metadata "$RESULTS/server.json" > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

python "$SCRIPTS/server.py" wait --base-url "$BASE_URL" --server-pid "$SERVER_PID"

python "$SCRIPTS/accuracy.py" run \
  --model "$MODEL" --base-url "$BASE_URL" --label inductor \
  --out "$RESULTS/accuracy_inductor" \
  --vllm-ascend-source "$PWD/vllm-ascend" --benchmark-home <AISBench目录> \
  --mmlu-dataset <MMLU数据集目录> --gsm8k-dataset <GSM8K数据集目录> \
  --tasks mmlu,gsm8k --num-prompts 200

python "$SCRIPTS/performance.py" run \
  --model "$MODEL" --base-url "$BASE_URL" \
  --dataset-name sharegpt --dataset-path <ShareGPT.json> \
  --request-rate 4 --out "$RESULTS/sharegpt_qps_4.json"

python "$SCRIPTS/performance.py" run \
  --model "$MODEL" --base-url "$BASE_URL" \
  --dataset-name random --random-input-len 8192 \
  --request-rate inf --out "$RESULTS/random_8192_1024.json"
```

`performance.py run` 使用以下默认值：

| 参数 | ShareGPT | random | 原因 |
|---|---|---|---|
| 生成参数 | `temperature=0`、忽略 EOS | 同左 | 固定生成工作量，避免文本和 EOS 差异改变输出 token 数 |
| 输出长度 | 256 tokens | 1024 tokens | ShareGPT 保留输入分布，同时固定输出长度；random 用于固定的长输出负载 |
| 请求数 | 200 | `4 × 最大并发` | 随机负载默认覆盖四轮，避免请求数变化导致测试波形不同 |
| 预热数 | 10 | 等于最大并发 | 预热覆盖目标并发，不把首次调度或图捕获影响带入正式结果 |
| 最大并发 | 不限制 | 8 | ShareGPT 模拟请求到达；random 使用固定并发测量吞吐和延迟 |

需要自然结束或非定长输出时，显式使用 `--no-ignore-eos`，并用 `--sharegpt-output-len`、`--random-output-len` 调整输出预算。`--request-rate` 始终显式指定，random 的 `--random-input-len` 也必须显式指定。服务默认使用 `--no-enable-prefix-caching`，避免重复提示词命中缓存；需要测试该特性时再单独开启。

TTFT 包含排队、prefill 和网络时间，不等于纯 prefill kernel 耗时。比较 TTFT 时必须保持请求数、并发和到达速率一致；需要定位 kernel 性能时另行 profiling。

每次性能调用只执行一种负载；不同 QPS、输入长度和重复轮次由外层显式组织。常规性能比较分别运行生产 default 与 Inductor，再比较相同负载的结果：

```bash
python "$SCRIPTS/accuracy.py" compare \
  --reference "$RESULTS/accuracy_eager/accuracy_results.json" \
  --candidate "$RESULTS/accuracy_inductor/accuracy_results.json" \
  --out "$RESULTS/accuracy_compare.json"
python "$SCRIPTS/performance.py" compare \
  --reference "$RESULTS/default/sharegpt_qps_4.json" \
  --candidate "$RESULTS/inductor/sharegpt_qps_4.json" \
  --out "$RESULTS/performance_compare.json" --require-within-thresholds
```

## 路径证据与汇总

```bash
python "$SCRIPTS/runtime.py" evidence \
  --run-config "$RESULTS/inductor/run_config.json" --log "$RESULTS/server.log" \
  --cache-dir "$RESULTS/cache/inductor" --artifact-dir "$RESULTS/inductor.compile_debug" \
  --out "$RESULTS/inductor_evidence.json" --require-inductor

python "$SCRIPTS/report.py" \
  "$RESULTS/generation_compare.json" "$RESULTS/accuracy_compare.json" \
  "$RESULTS/performance_compare.json" --out "$RESULTS/SUMMARY.md"
```

路径证据不替代数值正确性或图回放验收。每个 rank 的日志和缓存应分别传入、分别保存。

## AISBench 已知问题

`tools/aisbench.py` 会把日志中任何含 `ERROR` 的行判为失败。容器缺少 `mindstudio-probe`、`zstandard` 时，非致命 ERROR 可能导致预测完成但分数未生成。优先安装依赖：

```bash
pip install zstandard
pip install xgrammar==0.2.8
```

预测已完整落盘时，直接离线补算，不重跑服务或推理：

```bash
python "$SCRIPTS/accuracy.py" rescore \
  --run-dir "$RESULTS/accuracy_inductor" --task mmlu --num-prompts 200 \
  --out "$RESULTS/mmlu_rescore.json"
```

补算保留 `--num-prompts 200`，避免分母变回完整数据集；结果写入新文件，不改写原始结果。

## 单模型性能基准

`run_performance.sh` 将上述服务与性能命令组合为固定基准：ShareGPT QPS 1/4/16/inf 和 8192/1024、32768/1024 两组长上下文，均使用 `performance.py run` 的默认负载参数。

```bash
bash run_performance.sh \
  --model /mnt/weight/Qwen3.8-27B-w8a8 \
  --workspace <workspace绝对路径> \
  --results <新的结果根目录绝对路径> \
  --sharegpt <ShareGPT.json绝对路径> \
  --devices 0,1 --tp 2
```

脚本固定运行两轮，顺序为 `default → Inductor → default → Inductor`。每条路径复用自己的编译缓存；启动前检查 NPU 进程和 HBM 占用，并通过卡锁防止同一批次重复分配。每轮完成后自动生成 6 份比较 JSON，两轮共 12 份，最后生成并打印 `<结果根目录>/performance/SUMMARY.md`。

单卡模型使用 `--devices 0 --tp 1`；`--devices` 的数量必须等于 `--tp`。`--cache-root <目录>` 可复用该目录下的 `default/` 和 `inductor/` 暖缓存；不传时使用本次结果目录下的新缓存。`--timeout` 控制每次服务进程的最长运行时间，默认 21600 秒。结果根目录必须不存在，避免覆盖历史数据。

## 可选的完整矩阵编排

`run_matrix.sh` 是简单的三阶段批次脚本：先正确性，再任务集，最后性能。模型按清单顺序从小到大运行。

- 小模型使用 1 张卡，固定分到最多 4 个 worker；每个 worker 顺序处理自己的模型，卡被外部任务占用时等待。
- 大模型在小模型全部结束后顺序运行，按 TP2/4/8 选择空闲卡，不足就直接跳过。
- 空闲检查使用 `flock` 和 `npu-smi`；HBM 已用空间超过 `BENCH_MAX_USED_MB` 的卡不分配。
- 文本和多模态模型运行 MMLU、GSM8K 各 200 题，eager、Inductor 分别执行；Embedding 只跑正确性。
- 性能包含 ShareGPT QPS 1/4/16/inf 和两组长上下文，顺序为 `default → Inductor → default → Inductor`。
- 缓存保存在 `<模型>/cache/<eager|default|inductor>`，同一缓存复用于正确性、任务集和两轮性能。

```bash
BENCH_WORKSPACE=<workspace绝对路径> \
BENCH_AISBENCH_HOME=<AISBench目录> \
BENCH_MMLU_DATASET=<MMLU数据集目录> \
BENCH_GSM8K_DATASET=<GSM8K数据集目录> \
BENCH_SHAREGPT_PATH=<ShareGPT.json绝对路径> \
bash run_matrix.sh <新的结果根目录绝对路径>
```

结果根目录必须不存在。可选配置只有：`BENCH_DEVICES`（默认 0–7）、`BENCH_MODELS`（逗号分隔子集）、`BENCH_MAX_USED_MB`（默认 1024）和 `BENCH_TIMEOUT`（默认 21600 秒）。

每个模型保存 `correctness.log`、`accuracy.log`、`performance.log` 和对应结果目录。失败会打印在控制台并继续其他模型，最后以非零退出码汇总；大模型卡不足只打印 `SKIP`。脚本不包含设备池、状态机、自动报告或续跑逻辑。

## 旧入口

更复杂的远程连接、跨机并行、全局 deadline 和结果同步仍由 CI 或外层调度系统负责。本目录不恢复 `run_nightly.py`、`run_model.sh`、图模式总控及绑定固定目录的汇总入口。

部署仍遵循本地修改、git 管理、再同步到服务器的流程。整目录同步可以避免已删除的旧脚本继续留在服务器端。

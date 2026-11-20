# benchmark：vllm-ascend Inductor 接入测试脚本

本目录是 Inductor backend 接入 vllm-ascend 的**长期测试脚本**，git 维护，只放脚本与说明，不放测试结果。

- 测试规划、模型/量化矩阵与验收标准：上级目录《[Inductor适配测试覆盖模型与量化矩阵.md](../Inductor适配测试覆盖模型与量化矩阵.md)》
- 测试原始结果归档：本机 `D:\HuaweiWorkSpace\Inductor\server-results\benchmark\`（本地 `server-scripts/` 只是临时脚本区）

## 脚本清单

按流水线阶段命名：correctness（固定输入正确性）→ task accuracy（AISBench 任务分数）→ performance（官方服务性能）。

| 脚本 | 作用 |
|---|---|
| `run_nightly.py` | Nightly 总控（本地跑，ssh + docker exec 到服务器）：22 模型并行，正确性 → AISBench 任务精度（nightly 模型）→ 官方性能（default 与 inductor 两种编译路径），含设备池调度、deadline 控制、结果同步回本地并汇总 |
| `run_model.sh` | 单模型 × 多编译路径（eager/default/inductor）驱动器；`MODEL_SET=20261117\|20261118` 选模型集；负责缓存按编译路径隔离并记录 cold/warm、环境快照（git SHA、包版本、脚本 sha256）、flock 互斥 |
| `check_correctness.py` | 正确性检查主体（由 run_model.sh 调用）：8 组固定输入（4 短 4 长）× 2 轮、`max_tokens=1024`、`logprobs=5`、temperature=0，产出 `acc_track.jsonl`；输出与 eager 不一致时写 `review_required.json`（只标记，不中断） |
| `check_embedding.py` | Qwen3-Embedding-0.6B 专用：`llm.embed` 输出向量与 eager 比（余弦相似度、最大绝对差） |
| `compare_tracks.py` | 比较两个编译路径的 `acc_track.jsonl`，产出 `compare.json`（逐 token 一致数、首个分歧位置） |
| `check_graph_modes.sh` | 模型 1/12 的 5 种图模式（NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE）功能检查 + Qwen3-0.6B TP2 smoke |
| `run_task_accuracy.py` | AISBench 标准任务（MMLU/GSM8K 各 200 题，输出上限 2048）：起 OpenAI server，经 vllm-ascend `tools/aisbench.py` 包装器调 ais_bench |
| `run_serve_performance.py` | 官方 `vllm bench serve` 性能：ShareGPT 200 条 × QPS 1/4/16/inf + 8192/1024×32、32768/1024×16 长上下文，各 3 次重复 |
| `summarize_runs.py` / `summarize_nightly.py` | 汇总结果目录生成 Markdown 报告（前者对应 run_model.sh 的目录布局，后者对应 run_nightly.py 的三阶段布局） |
| `process_supervisor.py` | 进程组管理、致命错误检测与清理（run_model.sh 与性能/精度脚本调用） |

**改名对照**（2026-11-20，只改名未改行为；历史结果 `environment.txt` 中 sha256sum 记录的是旧文件名）：

| 旧名 | 新名 |
|---|---|
| `run_nightly_batch.py` | `run_nightly.py` |
| `run_a5.sh` | `run_model.sh` |
| `bench_one.py` | `check_correctness.py` |
| `embedding_one.py` | `check_embedding.py` |
| `run_graph_modes.sh` | `check_graph_modes.sh` |
| `run_standard_accuracy.py` | `run_task_accuracy.py` |
| `run_official_performance.py` | `run_serve_performance.py` |
| `summarize_matrix.py` | `summarize_runs.py` |

`compare_tracks.py`、`process_supervisor.py`、`summarize_nightly.py` 未改名。

## 运行方式

脚本实际在 A5 服务器的 CANN 容器内执行，本地这份是 git 维护的权威副本；改动流程：**本地改 → git 提交 → scp 同步到服务器**，不在服务器端直接改。

```bash
# 同步到服务器：整目录替换，先删远端旧目录，避免新旧两套脚本并存
# （服务器端脚本目录可自定义，当前为 <workspace>/scripts/benchmark_matrix，
# 历史上曾用名 bench213_matrix / bench213_matrix_rebase）
ssh <host> 'rm -rf <server-scripts-dir>' && scp -r . <host>:<server-scripts-dir>

# 单模型正确性（容器内）
cd <workspace> && MODEL_SET=20261118 TRACKS="eager inductor" \
  BENCH_DEVICES=2,3 bash run_model.sh /abs/result/root 模型名

# 图模式功能检查（容器内）
bash check_graph_modes.sh Qwen3-0.6B

# 远程批量 nightly（本地或跳板机；默认值按 a5-25 环境写死，可覆盖）
python run_nightly.py --run-id 20261120_nightly_01 \
  --local-root <本地结果目录> --deadline <ISO时间> --correctness-deadline <ISO时间>
```

路径说明：`run_model.sh`、`check_graph_modes.sh` 顶部硬编码了服务器绝对路径（`BASE`/`WORKSPACE`/`SCRIPTS`），换环境需同步修改；`run_nightly.py` 支持环境变量覆盖（`BENCH_HOST`、`BENCH_CONTAINER`、`BENCH_BASE`、`BENCH_WORKSPACE`、`BENCH_SCRIPTS`）。

## AISBench 标准任务（MMLU/GSM8K）注意事项

任务精度阶段通过 vllm-ascend 的 `tools/aisbench.py` 包装器调用 ais_bench。**已知坑：包装器的 `_check_runtime_error` 只要日志里出现任何含 `ERROR` 的行就判整个阶段失败；而 ais_bench 的响应异常检测（response_anomaly）功能在容器缺 `mindstudio-probe`、`zstandard` 两个可选依赖时会打印非致命 ERROR**（ais_bench 自己会继续打分，是被包装器拦腰砍断的）。后果：推理 200 题全部完成、预测 JSONL 已落盘，但分数没算、阶段被标失败。两种处理方式：

### 方式 A：跑之前把依赖装上（推荐）

```bash
# 容器内的 conda 环境里执行
pip install zstandard          # mindstudio-probe 若软件源里没有，见方式 B
pip install xgrammar==0.2.8    # 装上面的包可能顺带升 xgrammar，务必锁回 vllm 要求的版本
```

`mindstudio-probe` 只影响 response_anomaly 的准备阶段告警：装不上就还会剩一条 ERROR、任务精度阶段仍会被包装器中断，此时用方式 B 兜底。

### 方式 B：事后离线补算分数（不用重跑推理）

预测文件保留在 `accuracy/<模型>/<编译路径>/outputs/default/<时间戳>/predictions/` 下，在**该目录**里执行（容器内、conda 环境的 PATH）：

```bash
cd <结果目录>/accuracy/<模型>/<编译路径>
ais_bench --models vllm_api_general_chat_custom --datasets mmlu_200_custom \
  --mode eval --reuse <时间戳> --num-prompts 200
# 分数输出在 outputs/default/<时间戳>/summary/summary_*.csv
```

三个要点：

- **必须带 `--num-prompts 200`**，否则分母变成 mmlu miscellaneous 全集 783 题（实测会把 63.5% 算成 16.22%）
- `--datasets` 用裸名 `mmlu_200_custom`，不要写 `mmlu/mmlu_200_custom`（斜杠形式在当前 ais_bench 版本匹配不到配置）
- `--mode eval` 不跑推理也不跑 response_anomaly，不受缺依赖影响；`--reuse` 取指定时间戳的已有预测

实际案例：2026-11-20 run03 矩阵 10 个模型全部踩坑，事后用方式 B 补算 16 份预测，分数清单在 `server-results/benchmark/20261117_A5机器模测_Inductor矩阵/results/20261120_rebase_matrix_03/mmlu_rescore_results.txt`，结论见本目录《20261120_A5-28_rebase_matrix_03.md》。

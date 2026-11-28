---
name: full_benchmark
description: 在 A5 容器里用 tests/benchmark 脚本批量跑 vLLM Ascend 的 default 与 Inductor 性能对比（多模型并行、运行时动态选卡、DeepSeek 阶段、汇总报告），含开跑前重建 torch_npu 和核对缓存隔离。When to use: 用户要求跑一批模型的性能测试、Inductor 对比、捕获 1024 批次、cap1024、全量 benchmark，或提到 full_benchmark、性能批次、多卡并行跑模型时。
---

# full_benchmark：vLLM Ascend Inductor 性能批次

benchmark 本体脚本以 `doc/InfraDocs/TorchNPU_Inductor/tests/benchmark/` 为准（InfraDocs `feat/benchmark-capture-size-controls`，提交 `fe6461e` 及之后）。本目录 `scripts/` 只放批次调度和报告表格脚本，不要照搬到别的场景，按下面的步骤裁剪。

## 调用时先确认

用户没说明的才问，括号内是默认值；用户说了就以用户为准：

1. 跑哪些模型？（`scripts/jobs.txt` 里未注释的行：14 个 TP1 模型 + Qwen3.8-27B-W8A8 的 TP2；GLM 不跑；Qwen3.5-35B-A3B-w8a8-mtp 默认不跑，要跑就用 TP1）
2. 每个模型几轮？（2）3 轮约多 50% 时间。
3. 同时最多用几张卡？（4；用满 8 张需要用户明确同意）
4. 图捕获上限？（1024）
5. 其他模型结束后要不要跑 DeepSeek-V4.1-Flash-1017？（跑；只 1 轮；8 卡全空才跑；参数按官方 950DT DP8/TP1/EP8）
6. 有没有截止时间？（无；有则在会话内设一次性停止任务）
7. 批次名？（`perf_cap1024_<日期>_<序号>`）
8. 用户的 Inductor 源码有新提交时，装的是不是当前 HEAD？不是就先问是否重建安装（见 `scripts/rebuild_torch_npu.sh`）。

## 固定规则

**选卡**

- 不在计划阶段指定卡号；每个任务启动时才用 `runtime.py check-devices` 找空闲卡，拿到后用目录锁占住，结束释放。顺序按 7→0（空闲时优先后 4 张）。
- 并发任务合计不超过卡预算；TP2 要同时拿到 2 张卡；传给 vLLM 的卡列表升序。不抢占别人的卡，没空闲就等。

**开跑前核对**

- 安装的 torch_npu 必须等于源码 HEAD：`diff -rq site-packages/torch_npu/_inductor <源码树>/torch_npu/_inductor` 出现 `differ` 就重建安装（先构建成功，再卸旧装新，不 checkout 到别的提交）。
- 每个任务用结果目录下自己的空缓存；开跑后抽查运行中服务的 `/proc/<pid>/environ`，`VLLM_CACHE_ROOT`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR` 都要在该任务的 `perf/cache/<track>/` 下。
- 服务器上的脚本副本逐文件比 md5；模型路径、ShareGPT 数据集、磁盘空间先确认。
- 调度器改过就先空跑：`BATCH=<临时目录> DRY_RUN=10 bash ./run_all.sh`，看选卡、预算、TP2 升序、无残留锁。

**运行方式**

- 调度器由一个可跟踪的后台任务直接执行：`bash ./run_all.sh 2>&1 | tee -a "$BATCH/logs/master.log"`（先 `mkdir -p "$BATCH/logs"`）。每个任务的输出经 `tee` 同时写 `logs/perf_<名字>.log` 和调度器的标准输出（带任务名前缀），所以任务输出里实时能看到全部日志，`master.log` 是完整副本。不用脱离会话的进程，不另设监控或定时任务。
- `rc≠0` 或疑似卡死时看任务输出或 `status.sh`，向用户说明，不自行 kill；要停就 `stop.sh`（按进程树直接 SIGKILL，几秒完成）。
- 结束时确认 8 卡 HBM 回基线、无残留进程、`/dev/shm` 恢复 16G。

**失败处理**

- 单个负载请求失败或不完整：`run_performance.sh --load-retries`（默认 2）在同一服务上自动重跑，失败结果保留为 `<名字>.failed_attempt<k>.json`，不用管。
- 整个任务失败（服务起不来等）：看 `logs/perf_<名字>.log` 尾部，向用户说明现象并问是否用 `--cache-root` 复用缓存重跑；不要自动重跑整个模型。

**结果与报告**

- `tests/` 只放最终报告（`<日期>_<主题>_SUMMARY.md`），不放中间文件，不改用户的其他报告。
- 脚本放 `server-scripts/benchmark-runs/<日期_主题>/`，原始结果放 `server-results/benchmark/<日期_主题>/`（不含 cache，附脚本快照）。
- 报告结构：执行摘要、环境与缓存、方法、性能总览、编译缓存复用、与上一批对比（如有）、DeepSeek 章节（如跑）、完整性检查、文件位置、模型详细。总览表 >1.05 加粗、<0.95 斜体。对比不同轮数或不同 torch_npu 版本的批次，只写方向一致的结论。
- 语言按工作区 AGENTS.md：避免“对拍、口径、xx轨、钉xx、收口”。

## 脚本

| 文件 | 作用 | 说明 |
|---|---|---|
| `scripts/jobs.txt` | 任务清单，每行 `名字\|权重路径\|tp\|额外参数` | 跑哪些模型在这里改；`#` 开头的行忽略 |
| `scripts/common.sh` | 路径和开关 | 全部可用环境变量覆盖：`V`、`BATCH`、`SCRIPTS`（服务器上 benchmark 副本目录，默认 `scripts/benchmark_20261128_v9`）、`ROUNDS`（2）、`MAX_CARDS`（4）、`CAPTURE`（1024）、`DEEPSEEK`（1） |
| `scripts/run_all.sh` | 主调度器：读 jobs.txt、动态选卡、卡锁、预算，结束后调 `deepseek.sh`；任务输出经 `tee` 写各自日志并带前缀转到标准输出 | `BATCH=... MAX_CARDS=8 ROUNDS=2 bash ./run_all.sh 2>&1 \| tee -a "$BATCH/logs/master.log"` |
| `scripts/deepseek.sh` | DeepSeek 阶段：8 卡空闲检查、`/dev/shm` 临时扩到 512G 并恢复、三组参数阶梯、1 轮 | 模型路径和尝试参数按需改 |
| `scripts/status.sh` | 进度：已完成任务、卡锁、主日志尾、最近 30 分钟新文件数、每卡 HBM | — |
| `scripts/stop.sh` | 停批次：按脚本名匹配 `run_all.sh`/`run_performance.sh`/`deepseek.sh`（不看路径、不匹配自身），SIGKILL 整棵进程树，清卡锁，恢复 `/dev/shm`，几秒完成 | 会停容器里所有这样的批次 |
| `scripts/summarize.py` | compare JSON → Markdown + `summary.json` | `--entry NAME=PERF_DIR --rounds N`，有重试结果就用重试目录 |
| `scripts/make_report_tables.py` | 从 `summary.json` 生成总览、编译复用、与上一批对比、详细表 | 第二个参数是上一批 `summary.json`；本批路径在 `--batch` 下按 `<模型>/perf/performance` 找日志 |
| `scripts/rebuild_torch_npu.sh` | 从当前检出重建并安装 torch_npu | `TORCH_NPU_ARCH`、`--torch` 版本按环境改 |

## 运行手册

1. 确认容器、工作区、`env.sh`；`npu-smi info` 看 8 卡；核对模型路径、数据集、磁盘。
2. 本地改 `tests/benchmark/`（如有）并提交 InfraDocs，`scp` 到服务器新目录，逐文件比 md5（Windows 的 `md5sum` 输出带 `*`，比对时去掉）。
3. 核对 torch_npu 版本（见固定规则），需要就重建，之后 `diff` 为 0、`te_smoke.py` 通过再继续。
4. 改 `jobs.txt`，`scp` 本目录 `scripts/` 到服务器 `scripts/<主题>/`，比 md5，空跑验证。
5. 启动：把 `BATCH`、`ROUNDS`、`MAX_CARDS`、`CAPTURE`、`DEEPSEEK`、`SCRIPTS` 作为环境变量，用一个可跟踪的后台任务执行 `mkdir -p "$BATCH/logs" && bash ./run_all.sh 2>&1 | tee -a "$BATCH/logs/master.log"`。
6. 启动后 1–2 分钟核对 `JOB_LAUNCH` 的卡号、运行中服务的缓存环境变量、`environment.json` 里的 torch-npu 版本。
7. 进度看后台任务的输出；有截止时间就在会话内设一次性停止任务，到点执行 `stop.sh`。
8. 任务结束（`SCHEDULER_EXITED`）后：`summarize.py` → `make_report_tables.py` → 写报告；`tar --exclude='*/cache'` 拉回本地；确认服务器现场干净。
9. 收尾核对：条目 × 轮数 × 6 负载的 compare 齐全、捕获上限正确、Inductor 证据（backend=inductor、`mode=VLLM_COMPILE`、编译产物非零）、请求 `completed == num_prompts`、总览表标记存在。

## 已知情况

- 第 1 轮 Inductor 冷编译引擎初始化 61–207 秒，第 2 轮热加载 11–38 秒，日志含 `Directly load the compiled graph` 即复用成功。
- 35B 模型一轮约 30 分钟（ShareGPT qps1 固定约 200 秒）；8 卡并行、每模型 2 轮，15 个任务约 2 小时 50 分；4 卡并行、3 轮约 12 小时。
- ShareGPT 负载偶发 1/200 请求断连（`Connection reset by peer`），与代码无关，由 `--load-retries` 处理。
- DeepSeek-V4.1-Flash-1017 在 CANN 9.2.0 + vllm-ascend `0e82ba34` 的容器里：`/dev/shm` 扩到 512G 后 Engram 注册通过、8 卡各加载 45 GiB 权重，然后在第一次 dummy run 的 `MoeDistributeDispatchV3` tiling 报 `EZ1008 Get WinSize failed`；六组参数都一样，是环境阻塞。官方示例的 `--shm-size=512g` 必须满足，结束后恢复 16G。

## 坑

- 容器 bash 5.1 在 `set -u` 下，`declare -A X` 不赋值会把 X 当未定义，要写 `declare -A X=()`。
- `docker exec` 给 `bash -s` 喂脚本要加 `-i`。
- `npu-smi info` 的输出格式随终端宽度变化，HBM 用 `grep -oE '[0-9]+ +/ +98304'` 取，不按字段号。
- 容器时钟是 UTC、没有 zoneinfo，日志比北京时间慢 8 小时。
- 工具写文件落盘有延迟：同步前确认本地 md5 变了再 `scp`，同步后比两端 md5；heredoc 里的反斜杠序列会被改写，脚本内容尽量不含反斜杠。
- Windows 的 `tar` 把 `D:` 当远程主机，用 `/d/...` 路径并加 `--force-local`。
- `pgrep -f` 会匹配到外层 `bash -c` 包装进程，统计时按脚本名精确匹配。
- 工作区 `env.sh` 无条件 export 三个缓存变量，`runtime.py` 已兜底；最好把 `env.sh` 第 113–115 行改成 `${VAR:-}` 形式。

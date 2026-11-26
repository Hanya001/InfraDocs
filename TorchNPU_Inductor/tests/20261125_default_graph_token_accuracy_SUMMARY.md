# 2026-11-25 Inductor 默认图模式 token 精度测试报告

## 1. 测试范围与结论

日期：2026-11-25。

环境：a5-25 容器 `inductor_214build_z00896543_20261120_1546`，Ascend950DT，torch 2.13 / vLLM 0.33。

本轮依据《[Inductor适配测试覆盖模型与量化矩阵](Inductor适配测试覆盖模型与量化矩阵.md)》，使用 `benchmark/run_correctness.sh` 检查生成模型的 eager/Inductor 固定输入 token 和 logprob 精度。覆盖矩阵中的 21 个生成模型测试项；DeepSeek-V4.1 的 BF16 KV 和 FP8 KV 分开记录。Embedding 项不参与 token 生成测试。

结论：

- 14 个模型形成有效 eager/Inductor 对比，**logprob 判定全部未通过**。
- 模型级状态合计：`passed` 0、`failed` 15、`eager blocked` 3、`resource blocked` 3。
- 14 个有效对比共 112 个 prompt，其中 17 个 prompt 的 `pr_failure=null`。
- 8 个模型至少有 prompt 在 prefill 检查首先失败；另外 6 个模型的首个规则失败全部在 decode。
- 资源阻塞不是测试通过，也不是 eager/Inductor 数值失败。

## 2. 测试配置与判定规则

### 2.1 固定输入与生成配置

- 8 组固定文本输入：4 组短输入、4 组长输入；多模态模型本轮仅使用固定文本输入。
- eager 与 Inductor 各生成 2 轮，`temperature=0`、`seed=0`、`max_tokens=1024`、`ignore_eos=False`，保存 Top-20 logprobs。
- token 一致数按第 1 轮完整 token 序列统计；第 2 轮用于各路径自身的重复生成稳定性检查。
- 只使用默认 O2，不扫描其他图模式。14 个有效 Inductor 对比的最终图模式均为 `FULL_AND_PIECEWISE`，均有实际 backend 和编译产物证据。
- 所有已启动路径使用独立 cold 缓存；未运行任务集、性能或其他特性矩阵。

### 2.2 PR token/logprob 判定

以同权重、同量化、同 KV 精度的 eager 为参考：

1. 生成位置 0，即代码中的 prefill 检查：eager 与 Inductor 的首个输出 token 必须相同，logprob 绝对差不超过 0.0689。
2. 后续 decode 检查：logprob 绝对差不超过 0.1378；token 不同时，eager token 必须位于 Inductor 当前位置的 Top-20。
3. 输出 token 数也必须相同。
4. 8 个 prompt 全部满足，`logprob_within_tolerance` 才为 `true`。

阅读逐 prompt 结果时需注意：

- prompt 编号为 0–7：0–3 是短输入，4–7 是长输入。
- `pr_failure` 只记录每个 prompt 的**首个规则失败**，不是该 prompt 的全部失败清单。
- `first_diff_pos` 独立记录首次 token 分叉位置。首个 `pr_failure` 可能晚于 token 分叉；分叉后两边上下文已经不同，这些 decode 差值不是 teacher-forced 数值差，不能直接解释为根因。
- 下文“PR 规则通过”表示该 prompt 的 `pr_failure=null`，不等于整个模型通过。
- token 一致与 logprob 判定不是同一个条件：本轮存在完整 token 序列相同但 decode logprob 超差的 prompt。

## 3. 结果总表

| 模型 | 量化/KV | 拓扑 | token 一致 | logprob 判定 | 状态与说明 |
|---|---|---:|---:|---|---|
| Qwen3-0.6B | BF16 | TP1 | 2/8 | 未通过 | `failed` |
| Qwen3-0.6B-W8A8 | W8A8 Dynamic | TP1 | 0/8 | 未通过 | `failed` |
| Qwen3.5-0.8B | BF16 | TP1 | 5/8 | 未通过 | `failed` |
| Qwen2.5-VL-7B-Instruct | BF16，文本输入 | TP1 | 5/8 | 未通过 | `failed` |
| Qwen3-8B-W8A8 | 静态 W8A8 | TP1 | 0/8 | 未通过 | `failed` |
| DeepSeek-V2-Lite | BF16 | TP1 | 5/8 | 未通过 | `failed` |
| DeepSeek-V2-Lite-W8A8 | W8A8 Dynamic | TP1 | — | — | `eager blocked`：eager 引擎初始化失败，未运行 Inductor |
| Qwen3.8-27B | BF16 | TP1 | 1/8 | 未通过 | `failed` |
| Qwen3.8-27B-w8a8 | W8A8 | TP1 | 0/8 | 未通过 | `failed` |
| Qwen3.8-27B-FP8 | FP8 block | TP1 | — | — | `failed`：eager 完成，Inductor 编译后初始化失败，无有效对比 |
| Qwen3.8-27B-w8a8-mxfp8 | W8A8-MXFP8 | TP1 | 0/8 | 未通过 | `failed` |
| Qwen3-30B-A3B-W8A8 | W8A8 Dynamic | TP1 | 0/8 | 未通过 | `failed` |
| Qwen3.6-35B-A3B | BF16 | TP1 | 0/8 | 未通过 | `failed` |
| Qwen3.6-35B-A3B-w8a8 | W8A8 | TP1 | 1/8 | 未通过 | `failed` |
| Qwen3.6-35B-A3B-w4a8-mxfp | W4A8-MXFP | TP1 | 0/8 | 未通过 | `failed` |
| Qwen3.6-35B-A3B-w8a8-mxfp8 | W8A8-MXFP8 | TP1 | 0/8 | 未通过 | `failed` |
| Kimi-K3-expert16-top8-100B-CountBenchQA | BF16 | TP4 | — | — | `eager blocked`：eager 权重加载时 Worker 启动失败，未运行 Inductor |
| Qwen3.8-Flash-Next | BF16 | TP4 | — | — | `eager blocked`：eager 生成中 Worker 异常，未运行 Inductor |
| GLM-5.3-Flash-w8a8 | W8A8，计划 C8 KV | TP8 | — | — | `resource blocked`：NPU 7 有其他用户进程，未启动 |
| GLM-5.2-w4a8c8 | W4A8C8 | TP8 | — | — | `resource blocked`：NPU 7 有其他用户进程，未启动 |
| DeepSeek-V4.1-Flash-1017 | 原生 FP8/MXFP4，BF16 KV | TP8 | — | — | `resource blocked`：NPU 7 有其他用户进程，未启动 |
| DeepSeek-V4.1-Flash-1017 | 原生 FP8/MXFP4，FP8 KV | TP8 | — | — | `resource blocked`：NPU 7 有其他用户进程，未启动 |

`Qwen3-Embedding-0.6B` 为 Pooling/Embedding 模型，本轮明确只测生成 token，不建立 token 精度结果，也不将其计为通过。

## 4. prefill / decode 失败分类

下表按各 prompt 的首个 `pr_failure` 分类，不再只用“logprob 未通过”概括。

| 模型 | token 一致 | PR 规则通过 | prefill token 不同 | prefill logprob 超差 | decode logprob 超差 | decode token 不在 Top-20 |
|---|---:|---:|---|---|---|---|
| Qwen3-0.6B | 2/8 | 2/8 | 0 | 0 | 1：p1 | 5：p0、p3、p4、p6、p7 |
| Qwen3-0.6B-W8A8 | 0/8 | 0/8 | 4：p0、p1、p3、p5 | 3：p4、p6、p7 | 1：p2 | 0 |
| Qwen3.5-0.8B | 5/8 | 5/8 | 0 | 0 | 0 | 3：p5、p6、p7 |
| Qwen2.5-VL-7B-Instruct | 5/8 | 4/8 | 0 | 0 | 1：p6 | 3：p3、p5、p7 |
| Qwen3-8B-W8A8 | 0/8 | 0/8 | 5：p0、p1、p4、p5、p6 | 3：p2、p3、p7 | 0 | 0 |
| DeepSeek-V2-Lite | 5/8 | 5/8 | 0 | 0 | 2：p3、p7 | 1：p1 |
| Qwen3.8-27B | 1/8 | 1/8 | 0 | 1：p1 | 4：p2、p3、p4、p5 | 2：p0、p6 |
| Qwen3.8-27B-W8A8 | 0/8 | 0/8 | 1：p0 | 3：p1、p3、p4 | 4：p2、p5、p6、p7 | 0 |
| Qwen3.8-27B-W8A8-MXFP8 | 0/8 | 0/8 | 0 | 0 | 8：p0–p7 | 0 |
| Qwen3-30B-A3B-W8A8 | 0/8 | 0/8 | 0 | 3：p3、p5、p6 | 5：p0、p1、p2、p4、p7 | 0 |
| Qwen3.6-35B-A3B | 0/8 | 0/8 | 0 | 0 | 7：p1–p7 | 1：p0 |
| Qwen3.6-35B-A3B-W8A8 | 1/8 | 0/8 | 0 | 2：p0、p3 | 5：p1、p4、p5、p6、p7 | 1：p2 |
| Qwen3.6-35B-A3B-W4A8-MXFP | 0/8 | 0/8 | 1：p3 | 1：p1 | 5：p0、p4、p5、p6、p7 | 1：p2 |
| Qwen3.6-35B-A3B-W8A8-MXFP8 | 0/8 | 0/8 | 1：p3 | 2：p1、p2 | 3：p4、p5、p6 | 2：p0、p7 |

没有 prompt 以 `token_length_mismatch` 作为首个 PR 失败。

两个不能只看 token 一致数的例子：

- Qwen2.5-VL-7B-Instruct p6：完整 token 序列相同，但位置 526 的 decode logprob 差为 0.185865 > 0.1378。
- Qwen3.6-35B-A3B-W8A8 p1：完整 token 序列相同，但位置 12 的 decode logprob 差为 0.242018 > 0.1378。

## 5. 各模型逐 prompt 明细

### 5.1 Qwen3-0.6B

- PR 规则通过：2/8（p2、p5）；token 一致：2/8。
- 首次 token 分叉：p0@25、p1@8、p3@59、p4@36、p6@5、p7@45。
- `decode_logprob_mismatch`：p1@9，差值 0.192200 > 0.1378。
- `decode_token_not_in_candidate_topk`：p0@26（eager 311，Inductor 选择 3330）；p3@60（551 / 1039）；p4@37（105409 / 104396）；p6@6（624 / 14311）；p7@46（10693 / 279）。

### 5.2 Qwen3-0.6B-W8A8

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@0、p1@0、p2@57、p3@0、p4@5、p5@0、p6@9、p7@1。
- `prefill_token_mismatch`：p0（eager 444 → Inductor 8515）；p1（264 → 279）；p3（537 → 25383）；p5（51461 → 220）。
- `prefill_logprob_mismatch`：p4@0，0.163302 > 0.0689；p6@0，0.307039 > 0.0689；p7@0，0.407022 > 0.0689。
- `decode_logprob_mismatch`：p2@1，0.167986 > 0.1378。

### 5.3 Qwen3.5-0.8B

- PR 规则通过：5/8（p0–p4）；token 一致：5/8。
- 首次 token 分叉：p5@13、p6@89、p7@85。
- `decode_token_not_in_candidate_topk`：p5@14（eager 3709，Inductor 选择 96988）；p6@90（220 / 9333）；p7@86（9333 / 220）。

### 5.4 Qwen2.5-VL-7B-Instruct

- PR 规则通过：4/8（p0、p1、p2、p4）；token 一致：5/8。
- 首次 token 分叉：p3@64、p5@20、p7@31。
- `decode_logprob_mismatch`：p6@526，0.185865 > 0.1378。p6 没有 token 分叉，是完整 token 序列相同但 logprob 超差。
- `decode_token_not_in_candidate_topk`：p3@65（eager 10186，Inductor 选择 6974）；p5@21（101048 / 102445）；p7@32（315 / 86664）。

### 5.5 Qwen3-8B-W8A8

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@0、p1@0、p2@5、p3@4、p4@0、p5@0、p6@0、p7@20。
- `prefill_token_mismatch`：p0（eager 14511 → Inductor 8515）；p1（264 → 279）；p4（220 → 73157）；p5（101882 → 51461）；p6（576 → 220）。
- `prefill_logprob_mismatch`：p2@0，0.132269 > 0.0689；p3@0，0.249067 > 0.0689；p7@0，0.137251 > 0.0689。

### 5.6 DeepSeek-V2-Lite

- PR 规则通过：5/8（p0、p2、p4、p5、p6）；token 一致：5/8。
- 首次 token 分叉：p1@54、p3@110、p7@24。
- `decode_logprob_mismatch`：p3@111，0.589928 > 0.1378；p7@25，0.557278 > 0.1378。
- `decode_token_not_in_candidate_topk`：p1@55（eager 11，Inductor 选择 382）。

### 5.7 Qwen3.8-27B

- PR 规则通过：1/8（p7）；token 一致：1/8。
- 首次 token 分叉：p0@143、p1@48、p2@245、p3@142、p4@10、p5@220、p6@30。
- `prefill_logprob_mismatch`：p1@0，0.095963 > 0.0689。
- `decode_logprob_mismatch`：p2@245，0.314709 > 0.1378；p3@143，1.508892 > 0.1378；p4@12，0.176678 > 0.1378；p5@221，1.155237 > 0.1378。
- `decode_token_not_in_candidate_topk`：p0@144（eager 310，Inductor 选择 7129）；p6@31（6558 / 3018）。

### 5.8 Qwen3.8-27B-W8A8

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@0、p1@47、p2@179、p3@11、p4@8、p5@9、p6@2、p7@1。
- `prefill_token_mismatch`：p0（eager 8254 → Inductor 6584）。
- `prefill_logprob_mismatch`：p1@0，0.126826 > 0.0689；p3@0，0.126594 > 0.0689；p4@0，0.117273 > 0.0689。
- `decode_logprob_mismatch`：p2@2，0.220492 > 0.1378；p5@1，0.270187 > 0.1378；p6@1，0.243223 > 0.1378；p7@1，0.422628 > 0.1378。

### 5.9 Qwen3.8-27B-W8A8-MXFP8

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@39、p1@59、p2@392、p3@78、p4@19、p5@210、p6@30、p7@20。
- 8 个 prompt 的首个失败均为 `decode_logprob_mismatch`：p0@15（0.227000）；p1@8（0.147018）；p2@191（0.151150）；p3@8（0.223773）；p4@19（0.174297）；p5@72（0.157814）；p6@1（0.235281）；p7@1（0.223263）。容差均为 0.1378。

### 5.10 Qwen3-30B-A3B-W8A8

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@1、p1@11、p2@14、p3@14、p4@2、p5@11、p6@1、p7@31。
- `prefill_logprob_mismatch`：p3@0，0.158741 > 0.0689；p5@0，0.235558 > 0.0689；p6@0，0.330160 > 0.0689。
- `decode_logprob_mismatch`：p0@2，2.971161 > 0.1378；p1@1，0.149454 > 0.1378；p2@2，0.359970 > 0.1378；p4@2，0.260859 > 0.1378；p7@3，0.186850 > 0.1378。

### 5.11 Qwen3.6-35B-A3B

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@62、p1@20、p2@241、p3@321、p4@27、p5@39、p6@171、p7@95。
- `decode_logprob_mismatch`：p1@20（0.332222）；p2@60（0.155563）；p3@9（0.241152）；p4@17（0.149899）；p5@29（0.170313）；p6@172（22.439201）；p7@96（8.343433）。容差均为 0.1378。p6/p7 的极大差值发生在已有分叉的上下文中，不能直接作为算子根因。
- `decode_token_not_in_candidate_topk`：p0@63（eager 369，Inductor 选择 424）。

### 5.12 Qwen3.6-35B-A3B-W8A8

- PR 规则通过：0/8；token 一致：1/8。
- 首次 token 分叉：p0@6、p2@7、p3@6、p4@19、p5@5、p6@30、p7@95。
- `prefill_logprob_mismatch`：p0@0，0.077081 > 0.0689；p3@0，0.194323 > 0.0689。
- `decode_logprob_mismatch`：p1@12，0.242018 > 0.1378；p4@2，0.147242 > 0.1378；p5@5，0.166513 > 0.1378；p6@3，0.147036 > 0.1378；p7@37，0.172321 > 0.1378。p1 是 token 完全一致但 logprob 超差。
- `decode_token_not_in_candidate_topk`：p2@8（eager 3712，Inductor 选择 57902）。

### 5.13 Qwen3.6-35B-A3B-W4A8-MXFP

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@7、p1@12、p2@7、p3@0、p4@25、p5@10、p6@2、p7@54。
- `prefill_token_mismatch`：p3（eager 264 → Inductor 9613）。
- `prefill_logprob_mismatch`：p1@0，0.130833 > 0.0689。
- `decode_logprob_mismatch`：p0@5，0.192492 > 0.1378；p4@18，0.184338 > 0.1378；p5@9，0.187798 > 0.1378；p6@2，0.219069 > 0.1378；p7@2，0.203911 > 0.1378。
- `decode_token_not_in_candidate_topk`：p2@8（eager 57902，Inductor 选择 3712）。

### 5.14 Qwen3.6-35B-A3B-W8A8-MXFP8

- PR 规则通过：0/8；token 一致：0/8。
- 首次 token 分叉：p0@11、p1@20、p2@37、p3@0、p4@18、p5@20、p6@30、p7@2。
- `prefill_token_mismatch`：p3（eager 9613 → Inductor 1532）。
- `prefill_logprob_mismatch`：p1@0，0.144352 > 0.0689；p2@0，0.080252 > 0.0689。
- `decode_logprob_mismatch`：p4@10，0.310612 > 0.1378；p5@20，0.406248 > 0.1378；p6@29，0.180843 > 0.1378。
- `decode_token_not_in_candidate_topk`：p0@12（eager 321，Inductor 选择 353）；p7@3（eager 248069，Inductor 选择 8160）。

## 6. 无逐 prompt 对比的项目

- Qwen3.8-27B-FP8：Inductor 初始化失败，未生成候选输出，因此没有 `pr_failure`。
- DeepSeek-V2-Lite-W8A8、Kimi K3、Qwen3.8-Flash-Next：eager 未完成，未形成对比文件。
- GLM-5.3-Flash、GLM-5.2、DeepSeek-V4.1：资源阻塞，未启动，不存在 `pr_failure`。DeepSeek-V4.1 的 BF16 KV 和 FP8 KV 两条路径均未运行。

## 7. 权重、缓存与归档

- Qwen3-0.6B BF16 的实际权重路径为 `/mnt/weight/Qwen/Qwen3-0.6B`；27B W8A8-MXFP8 为 `/mnt/weight/weights/Qwen3.8-27B-w8a8-mxfp8`。
- Kimi K3 的 BF16 主索引及 39 个分片完整；目录中另有缺少分片的旧量化索引，未用于本次 BF16 加载。单独检查见原始归档中的 `weight_checks/Kimi-K3-expert16-top8-100B-CountBenchQA.bf16_main.json`。
- 14 个有效对比的两轮 eager 和 Inductor 生成均满足自身 token 序列稳定；这不改变 eager/Inductor 之间的 logprob 未通过结论。
- 每个已运行模型保存 `environment.json`、逐路径 runtime/run_config、token、文本、Top-20 logprobs、compare 和驱动日志。大体积编译缓存保留在服务器结果目录，本地不同步缓存文件。
- 完整的 Top-20 候选、每个位置的 logprob 和 `divergence` 信息保存在各模型 `correctness/compare.json` 与 `acc_track.jsonl` 中。

原始结果：

- 本地归档：`server-results/benchmark/20261125_token_01/`。
- 服务器归档：`/home/z00896543/validation/214build_20261120_1546/results/token_20261125_01/`。

归档检查确认：14 份有效 compare、33 份 runtime 记录和 3 份资源阻塞记录齐全。测试结束时未遗留本次启动的测试或监控进程；未终止其他用户的 NPU 7 进程。

## 8. 运行失败项重试复核

随后在相同容器中对 4 个非资源阻塞模型进行了独立 cold 缓存重试。当前 meta、pytorch、vllm-ascend、triton-ascend 提交，软件版本和 `env.sh` 哈希均与首轮相同；没有证据表明故障已修复，因此本轮用于确认故障是否仍存在，不覆盖首轮记录。

| 模型 | 重试配置 | 重试结果 |
|---|---|---|
| Qwen3.8-27B-FP8 | TP1，NPU 0 | 仍为 `failed`。eager 完成，Inductor 初始化再次报 `EngineCore failed to start`，无有效 token 对比 |
| DeepSeek-V2-Lite-W8A8 | TP1，NPU 1 | 仍为 `eager blocked`。权重加载完成、KV Cache 初始化后再次报 `EngineCore failed to start`，未进入 Inductor |
| Kimi-K3-expert16-top8-100B-CountBenchQA | TP4，NPU 0–3 | 仍为 `eager blocked`。eager 权重加载中再次出现 `WorkerProc failed to start` |
| Qwen3.8-Flash-Next | TP4，NPU 0–3 | 仍为 `eager blocked`。权重加载完成、KV Cache 分组阶段出现 `WorkerProc hit an exception`；本次未进入生成，未运行 Inductor |

结论：4 个模型的运行故障均可复现，当前仍不能形成完整 eager/Inductor 精度对比。三个 TP8 模型没有重试；复核时 NPU 0–6 空闲，NPU 7 仍有其他用户进程，资源阻塞状态不变。

重试原始结果：

- 本地归档：`server-results/benchmark/20261125_token_retry_01/`。
- 服务器归档：`/home/z00896543/validation/214build_20261120_1546/results/token_retry_20261125_01/`。

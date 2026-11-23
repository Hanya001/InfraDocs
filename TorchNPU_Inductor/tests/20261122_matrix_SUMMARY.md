# Inductor 生成模型矩阵测试汇总

日期：2026-11-22。环境：a5-25，Ascend950DT，torch 2.13 / vLLM 0.33。

## 范围、判据与停止状态

- 矩阵第 19 项 Embedding 不在本次目标内；多模态模型只执行文本输入。21 个生成模型测试项均有独立记录，其中 16 项实际启动，5 个大模型项按用户要求暂停并标记 `skipped`。
- 每个已执行模型先运行 eager 固定输入；DeepSeek-V2-Lite-W8A8 为 `eager blocked`，后续跳过。其余 15 个模型均形成 Inductor 固定输入/logprob 对比，**没有任何一项同时满足 logprob 容差**。
- logprob 容差：平均绝对差不超过 0.05、最大绝对差不超过 0.5。总体状态为 `failed` 不一定表示服务无法运行，多数模型是数值或性能未通过。
- MMLU/GSM8K 通常各固定 200 题，27B-W8A8、35B-W8A8 使用固定 1000/500 题。GPQA 有效配置为全量 198 题、输出上限 32768；旧 2048 结果均无效。
- 用户停止剩余 GPQA 和大模型测试后，不再启动新的模型任务。停止前的部分结果仅用于溯源，不形成完整对比。
- 性能加速比 >1.05 使用**粗体**，<0.95 使用*斜体*；等于边界时不标记。
- 性能统一使用 Inductor 相对 default 的加速比，均为越高越好：吞吐=`Inductor/default`，TPOT/TTFT=`default/Inductor`。阈值为吞吐加速比 >=0.95、TPOT 加速比 >=1/1.05、TTFT 加速比 >=1/1.10。逐负载原始数值见 [PERFORMANCE_DETAIL.md](PERFORMANCE_DETAIL.md)。
- 同一模型后续阶段复用对应隔离缓存；运行记录保存 cold/warm 状态。`warm` 只表示复用缓存目录，不表示所有产物均命中。
- 所有自有测试和监控进程已清理。最后检查时 NPU 0、1、2、7 空闲，其他卡上的任务未触碰。

## 21 项结果总表

| 矩阵项 | 模型及记录 | 总体状态 | 固定输入/logprob：eager → Inductor | MMLU / GSM8K：eager → Inductor | GPQA 状态 | 性能状态 |
|---:|---|---|---|---|---|---|
| 1 | [Qwen3.5-0.8B](Qwen3.5-0.8B/20261122_RESULT.md) | `failed` | token 3/8；平均差 0.05439，最大差 0.62471 | 57.0→52.0（-5.0 pp）/ 52.5→55.5（+3.0 pp） | 不适用 | 8K 吞吐加速比 *0.840*、TPOT 加速比 *0.8264*；QPS1 TPOT 加速比 0.9520；复测 SIGKILL/137，无新数据 |
| 2 | [Qwen3.8-27B BF16](Qwen3.8-27B/20261122_RESULT.md) | `failed` | token 3/8；平均差 0.04747，最大差 0.93727 | 87.0→88.5（+1.5 pp）/ 95.5→95.0（-0.5 pp） | `skipped/incomplete`：Inductor 198/198；eager 补跑被用户停止，部分结果无效 | 6 组均在阈值内 |
| 3 | [Qwen3.8-27B-W8A8](Qwen3.8-27B-W8A8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.33818，最大差 2.88858 | 86.8→87.6（+0.8 pp，1000 题）/ 95.4→95.2（-0.2 pp，500 题） | `skipped/incomplete`：eager 83.84、198/198；Inductor 被用户停止 | 两轮共 12 组均在阈值内 |
| 4 | [Qwen3.8-27B-FP8](Qwen3.8-27B-FP8/20261122_RESULT.md) | `failed` | eager 完成；Inductor 冷/暖启动均失败，无有效对比 | eager 88.5 / 93.5；Inductor 未启动 | 旧 eager 35.35 为无效 2048 配置；Inductor 未执行 | 未执行有效对比 |
| 5 | [Qwen3.6-35B-A3B BF16](Qwen3.6-35B-A3B/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.09612，最大差 2.49999 | 78.0→77.5（-0.5 pp）/ 82.5→80.5（-2.0 pp） | 82.83→80.30（-2.53 pp），双路径 198/198 | 6 组均在阈值内 |
| 6 | [Qwen3.6-35B-A3B-W8A8](Qwen3.6-35B-A3B-W8A8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.16664，最大差 2.75000 | 73.3→74.0（+0.7 pp，1000 题）/ 84.4→83.0（-1.4 pp，500 题） | `failed`：两次 TP2 启动失败，无底层异常栈 | 两轮共 12 组均在阈值内 |
| 7 | [Qwen3.6-35B-A3B-W4A8-MXFP](Qwen3.6-35B-A3B-W4A8-MXFP/20261122_RESULT.md) | `failed` | token 1/8；平均差 0.22185，最大差 2.43744 | 72.5→72.5 / 80.0→74.5（-5.5 pp） | `skipped/incomplete`：首轮有失败请求，补跑被用户停止 | 首轮 8K TTFT 加速比 *0.8821* 低于阈值；其余通过；复测无新数据 |
| 8 | [Qwen3.6-35B-A3B-W8A8-MXFP8](Qwen3.6-35B-A3B-W8A8-MXFP8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.17578，最大差 2.24996 | 72.5→78.5（+6.0 pp）/ 78.5→80.0（+1.5 pp） | 82.32→81.31（-1.01 pp），双路径 198/198 | 首轮 8K TTFT 加速比 *0.8819* 低于阈值；其余通过；复测无新数据 |
| 9 | [DeepSeek-V4.1-Flash](DeepSeek-V4.1-Flash/20261122_RESULT.md) | `skipped` | 用户暂停，未执行 | 未执行 | 不适用 | 未执行；BF16/FP8 KV 均无结果 |
| 10 | [GLM-5.3-Flash-W8A8C8](GLM-5.3-Flash-W8A8/20261122_RESULT.md) | `skipped` | 用户暂停，未执行 | 未执行 | 不适用 | 未执行 |
| 11 | [Qwen3.8-Flash-Next](Qwen3.8-Flash-Next/20261122_RESULT.md) | `skipped` | 用户暂停，未执行 | 未执行 | 不适用 | 未执行 |
| 12 | [Qwen3-0.6B BF16](Qwen3-0.6B/20261122_RESULT.md) | `failed` | token 2/8；平均差 0.04922，最大差 1.62483 | 52.0→55.5（+3.5 pp）/ 70.5→67.5（-3.0 pp） | 不适用 | QPS1/QPS4 TPOT 加速比 *0.9184*/*0.8744*；暖、冷复测均启动失败 |
| 13 | [Qwen3-0.6B-W8A8](Qwen3-0.6B-W8A8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.27756，最大差 2.10955 | 43.0→50.0（+7.0 pp）/ 51.0→45.0（-6.0 pp） | 不适用 | QPS1 TPOT 加速比 *0.1583*、TTFT 加速比 *0.0733*；复测无新数据 |
| 14 | [Qwen3-8B-W8A8](Qwen3-8B-W8A8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.74411，最大差 5.74999 | 80.0→82.5（+2.5 pp）/ 88.5→89.5（+1.0 pp） | 不适用 | 6 组均在阈值内 |
| 15 | [Qwen3-30B-A3B-W8A8](Qwen3-30B-A3B-W8A8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.32717，最大差 4.125 | 85.0→85.5（+0.5 pp）/ 89.5→95.5（+6.0 pp） | 不适用 | 6 组均在阈值内 |
| 16 | [DeepSeek-V2-Lite BF16](DeepSeek-V2-Lite/20261122_RESULT.md) | `failed` | token 6/8；平均差 0.02757，最大差 2.59367 | 26.0→29.0（+3.0 pp）/ 11.0→12.5（+1.5 pp） | 不适用 | 6 组均在阈值内 |
| 17 | [DeepSeek-V2-Lite-W8A8](DeepSeek-V2-Lite-W8A8/20261122_RESULT.md) | `eager blocked` | eager 初始化失败，未完成生成；Inductor 跳过 | 跳过 | 不适用 | 跳过 |
| 18 | [Qwen2.5-VL-7B-Instruct（文本）](Qwen2.5-VL-7B-Instruct/20261122_RESULT.md) | `failed` | token 4/8；平均差 0.04069，最大差 1.70880 | 75.0→76.5（+1.5 pp）/ 87.0→85.5（-1.5 pp） | 不适用；TextVQA 不在本轮文本范围 | QPS1、QPS4、32K 超阈值；复测无新数据 |
| 20 | [GLM-5.2-W4A8C8](GLM-5.2-W4A8C8/20261122_RESULT.md) | `skipped` | 用户暂停，未执行 | 未执行 | 不适用 | 未执行 |
| 21 | [Qwen3.8-27B-W8A8-MXFP8](Qwen3.8-27B-W8A8-MXFP8/20261122_RESULT.md) | `failed` | token 0/8；平均差 0.145996，最大差 2.81144 | 89.5→92.5（+3.0 pp）/ 93.5→94.5（+1.0 pp） | 85.35→82.83（-2.52 pp），双路径 198/198 | 首轮 8K TTFT 加速比 *0.9069* 低于阈值；其余通过；复测无新数据 |
| 22 | [Kimi-K3-expert16-top8-100B-CountBenchQA](Kimi-K3-expert16-top8-100B-CountBenchQA/20261122_RESULT.md) | `skipped` | eager 启动后被用户停止，未完成生成；不是 eager blocked | 未执行 | 不适用 | 未执行 |

## 图模式与 Inductor 路径证据

- Qwen3.5-0.8B、Qwen3-0.6B 已覆盖 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`；五种模式均有生成及编译/捕获证据，但均未通过 logprob 容差。
- 其他已形成 Inductor 结果的模型保存了实际 backend、O2、图模式、服务日志和编译产物。启动失败或用户暂停的项目没有路径通过证据。
- 编译产物检查显示，量化模型仍调用 CANN 量化 GEMM、SwiGLU、NormQuant，额外 Triton kernel 主要涉及 RMSNorm 和类型转换；不能据此把两条路径的数值差异归因于共享算子。

## 性能异常与复测结论

- Qwen3.5-0.8B：8K 吞吐和 TPOT 明显劣化；追加测量进程 SIGKILL/137。
- Qwen3-0.6B：QPS1/QPS4 TPOT 劣化；暖、冷缓存复测均在 default 初始化失败。
- Qwen3-0.6B-W8A8：QPS1 TPOT/TTFT 明显劣化；复测 default 初始化失败。
- Qwen2.5-VL：QPS1/QPS4/32K 超阈值；复测 default 初始化失败。
- 27B-W8A8-MXFP8、35B-W4A8-MXFP、35B-W8A8-MXFP8：8K TTFT 临界或明显超阈值；复测 default 初始化失败。两个 35B 变体去掉首批后 TTFT 加速比仍约 0.8772，不是仅由首批冷编译造成。
- 上述复测均未产生新数据，本汇总保留首轮真实结果，不写成“复测通过”，也未继续无新依据地重启。

## 工具、原始结果与后续建议

- 本轮完善了 GPQA、专家并行、KV dtype、tokenizer mode、quantization、trust-remote-code、上下文长度和显存参数传递；工具改动位于 `doc/InfraDocs/TorchNPU_Inductor/tests/benchmark/`，尚未提交。
- 没有保留本轮模型或 Inductor 精度修复。35B-W8A8 记录中提到的 `npu_grouped_matmul_swiglu_quant_v2` 为此前已有启动修复。
- 每项目录下的 `20261122_RESULT.md` 给出权重、配置、缓存、原始日志位置和未解决项。27B-W8A8、35B-W8A8 的同日历史结果已同步到本批次对应目录，不再引用不存在的本地路径。
- 后续优先处理所有已运行模型共性存在的 logprob 超容差问题；性能问题需要先恢复可复现的 default 对照服务。大模型及未完成 GPQA 只有在用户明确恢复后才继续。

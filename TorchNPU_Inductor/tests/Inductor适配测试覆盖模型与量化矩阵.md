# vLLM Ascend Inductor 接入测试覆盖模型与量化矩阵

## 1. 目的与测试规则

本文定义 Inductor backend 接入 vllm-ascend 时的模型、量化版本、特性分工、CI 分层和验收判据。目标是同时覆盖：

- 新模型及新架构主路径。
- 仍需兼容的旧架构。
- 新模型和旧模型的代表性量化版本。
- 正确性、性能以及原有后端不回归。

本文是**测试规划和支持范围定义，不是测试通过记录**。每个场景的实际结果必须单独记录，不能由模型文档、测试文件存在或其他架构通过外推。

统一遵循以下规则：

1. 不做模型、量化、图模式和特性的全笛卡尔积。小模型跑完整编译矩阵，中模型跑主要特性，大模型跑默认配置和架构专有路径。
2. 每一行均使用**同一权重、同一输入和同一运行参数**对比 Inductor 与 eager，不能用量化 Inductor 对比 BF16 eager。KV Cache 量化路径同理：FP8 KV 的 Inductor 只能对比同 FP8 KV 的 eager。
3. 性能除保留 eager 数据外，必须与当前生产默认后端进行同日、同环境、同负载比较。
4. 只将 eager 已通过的场景纳入 Inductor 通过率。eager 失败记为 `eager blocked`，不能记为 Inductor 通过。
5. P0 是 release 必测项，P1 是架构兼容或量化专项回归项；P0/P1 不等于 PR/nightly 的执行频率。
6. 实际执行时记录权重来源、`config.json` 中的架构和量化字段、软件版本、硬件、拓扑、缓存状态及测试证据。执行前先核对权重仓库真实存在、名称与本文一致。
7. 310P 不在 Inductor 支持范围内，显式排除，不建立测试项，也不得默认视为支持。
8. vllm-ascend 官方教程已发布的 AISBench 精度参考值（Qwen3.8-27B BF16/w8a8/w8a8-mxfp8、Qwen3.6-35B-A3B-w8a8 等）直接引用为 eager 基线起点，不重复建设已有基线。

## 2. 新模型主验收清单

以下 11 项构成本次新模型和新量化的主验收范围。

| 编号 | 模型及权重版本 | 量化版本 | 覆盖定位 | 级别 |
|---|---|---|---|---|
| 1 | `Qwen/Qwen3.5-0.8B` | BF16 | 新一代小模型；混合注意力（GDN+全注意力）、原生多模态、MTP；承担完整图模式矩阵 | P0 |
| 2 | `Qwen/Qwen3.8-27B` | BF16 | 新一代 Dense 主模型；混合注意力、原生多模态、262K 上下文 | P0 |
| 3 | `Eco-Tech/Qwen3.8-27B-w8a8` | W8A8 | 新 Dense 的 ModelSlim 量化主路径；官方教程已有 GPQA 参考值，直接引用为基线 | P0 |
| 4 | `Qwen/Qwen3.8-27B-FP8` | FP8 block | 新模型原生 FP8 权重和 block scale 路径 | P0 |
| 5 | `Qwen/Qwen3.6-35B-A3B` | BF16 | 新一代中等规模混合注意力 MoE 主模型（256 专家，8 路由 + 1 共享） | P0 |
| 6 | `Eco-Tech/Qwen3.6-35B-A3B-w8a8` | W8A8 | 新 MoE 的 W8A8 主路径；官方教程已有 AISBench 参考值 | P0 |
| 7 | `Eco-Tech/Qwen3.6-35B-A3B-w4a8-mxfp` | W4A8-MXFP | 新 MoE 低比特权重、EP/TP、MTP 组合；注意 Eco-Tech 实际仓库名带 `-mxfp` 后缀 | P0 |
| 8 | `Eco-Tech/Qwen3.6-35B-A3B-w8a8-mxfp8` | W8A8-MXFP8 | 新 MoE 的 MXFP8 量化 kernel 和路由精度； | P0 |
| 9 | `deepseek-ai/DeepSeek-V4.1-Flash` | 原生 FP8 dense + MXFP4 路由专家 | 新 DSA/mHC/MoE 主模型；覆盖 MTP/DSpark、MRV2、多卡和 PD | P0 |
| 10 | `Eco-Tech/GLM-5.3-Flash-w8a8`（开启 C8 KV） | W8A8C8 | GLM 新主模型（GLM-5.3-Flash，320B-A18B，MLA+线性注意力混合，288 路由专家）；覆盖权重量化、C8 KV、DSA/SFA 和多机部署； | P0 |
| 11 | `Qwen/Qwen3.8-Flash-Next` | BF16 | Qwen4Exp 新架构预览（Qwen4 试验架构）；GDN + Qwen Sparse Attention 稀疏注意力、125B-A6B MoE、MTP；与 Qwen3.5/3.8 不同架构，不能被替代； | P1 |

DeepSeek-V4.1-Flash 需要分别记录 BF16 KV 和 FP8 KV 的结果。它们属于同一权重下的两个 KV 路径，不扩展成额外模型组合；仅验收 eager 已支持的路径。

## 3. 旧模型兼容与量化回归清单

### 3.1 P0 旧模型基础兼容项

| 编号 | 模型及权重版本 | 量化版本 | 保留理由 | 级别 |
|---|---|---|---|---|
| 12 | `Qwen/Qwen3-0.6B` | BF16 | 传统 Dense 小模型；保留旧架构和基础编译回归 | P0 |
| 13 | `Qwen/Qwen3-0.6B-W8A8` | W8A8 Dynamic | 传统 Dense 动态量化；已有 Inductor 测试和数值判据 | P0 |
| 14 | `vllm-ascend/Qwen3-8B-W8A8` | 静态 W8A8 | 静态量化、Norm+Quant 融合和 NZ 布局回归 | P0 |

### 3.2 P1 架构与量化专项

| 编号 | 模型及权重版本 | 量化版本 | 保留理由 | 级别 |
|---|---|---|---|---|
| 15 | `vllm-ascend/Qwen3-30B-A3B-W8A8` | W8A8 Dynamic | 传统 Full Attention MoE；兼顾 EPLB、TP 和旧 MoE 量化路径 | P1 |
| 16 | `deepseek-ai/DeepSeek-V2-Lite` | BF16 | 经典 MLA+MoE 架构回归，不能被 DSA 模型替代 | P1 |
| 17 | `vllm-ascend/DeepSeek-V2-Lite-W8A8` | W8A8 Dynamic | 经典 MLA 量化及动态量化外部算子路径 | P1 |
| 18 | `Qwen/Qwen2.5-VL-7B-Instruct` | BF16 | 传统 ViT encoder+LLM，多模态路径不同于 Qwen3.5/3.8 | P1 |
| 19 | `Qwen/Qwen3-Embedding-0.6B` | BF16 | 非生成式 Pooling/Embedding 接口回归 | P1 |
| 20 | `GLM-5.2-W4A8C8` | W4A8C8 | 与 GLM-5.3-Flash 基座相近，但承担独立的 W4A8C8 量化回归 | P1 |
| 21 | `Qwen/Qwen3.8-27B-w8a8-mxfp8` | W8A8-MXFP8 | 新 Dense 的 MXFP8 Linear/kernel 和数值路径；复用官方 AISBench 基线，不重复完整特性矩阵 | P1 |

旧模型只跑默认推荐配置、关键架构特性和必要的量化组合，不重复新模型的完整特性矩阵。

## 4. 特性与测试项的固定分工

以下分工直接指定到第 2、3 节编号，不再额外做模型和特性的自由组合。

| 特性 | 指定测试项 | 覆盖要求 |
|---|---|---|
| 全部图模式 | 1、12 | `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE` 和默认 `-O` 预设 |
| 默认图模式 | 2–11、13–21 | 验证默认推荐配置；另对高风险路径补 `PIECEWISE` 或 `FULL_DECODE_ONLY` |
| MRV2 | 5–11 | 新 MoE、DeepSeek、GLM、Flash-Next 以 MRV2 为主验收执行层 |
| 旧执行层兼容 | 12–17 | 保留仍声明支持的旧执行层回归，不要求旧模型同时跑所有 runner |
| 多模态 | 1、2、3、4、18 | 纯文本、单图、多图、不同分辨率、图文混合 batch；2–4 增加视频；18 验证传统 encoder |
| MTP/DFlash2 | 1–8、11 | 按模型实际提供的 draft 权重验证；同时检查 target 和量化 draft 的组合 |
| DSpark | 9、10、20 | 检查输出、接受率、图回放和 prefix cache 组合 |
| ngram | 12 | 保留通用非模型 draft 的推测解码回归 |
| Prefix Cache、Chunked Prefill | 1–11、13–17 | 新模型重点检查混合 KV、量化 KV、MTP 和图回放组合 |
| LoRA | 1、5、12 | 分别覆盖新小模型、新 MoE 和传统 Dense；包含单 LoRA、Multi-LoRA 和运行时加载 |
| LoRA × 量化 | 14 | 静态 W8A8 模型挂 LoRA，验证量化 kernel 与 LoRA 分支同时生效且数值正确 |
| 异步调度 | 2、5、9 | 默认开启路径下验证图回放、采样 RNG、连续请求一致性；与 prefix cache、MTP 组合 |
| 编译失败与回退 | 1、12 | 构造含不支持算子或强制 graph break 的配置，验证按预期回退且日志明确记录，不静默改变精度模式 |
| 结构化输出 | 1、12 | 约束采样与编译、图回放组合的输出合法性 |
| TP/EP/EPLB | 5–8、9、10、11、15 | Qwen MoE 覆盖 TP2/TP4 和 EPLB；DeepSeek/GLM 按推荐拓扑覆盖 TP/EP |
| PCP/DCP/PD | 9、10、20 | 验证真实 KV 传输、D 侧不重算 prefill、长序列和异常恢复 |
| 长序列 | 2–11、20 | 常规 32K；按模型支持边界执行 128K/256K；1M 仅作为明确支持模型（9、10）的专项 |
| Pooling/Embedding | 19 | 不同输入长度和 batch，对比 eager 的向量或分数 |
| 在线服务 | 1、2、5、9、10 | OpenAI 接口、连续请求、并发、thinking、tool calling 和服务重启 |
| 编译缓存 | 1、3、5、9、12、13、14 | 冷启动、暖启动、rank 隔离、配置 key 隔离、损坏缓存和重编译 |

DeepSeek-V4.1 等可能受 Breakable CUDAGraph 影响的模型，必须确认实际编译未被打断。需要显式设置 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 的场景，应在测试配置和结果中记录。

## 5. 量化验收要求

量化测试不能只检查加载和非空生成，必须包含以下判据：

1. **权重真实性**：执行前核对权重仓库存在、名称与本文一致、`config.json` 的量化字段与实际文件格式一致，防止测试名称与实际路径不符。
2. **数值正确性**：与同权重、同量化、同 KV 精度的 eager 做逐 token 或 logprob 对拍，并执行任务精度评估。FP8 KV 的结果只能对比 FP8 KV 的 eager，不能对比 BF16 KV 的 eager。
3. **真实量化路径**：检查权重格式、Dynamic/Static、量化算子和生成代码，防止走了 BF16 回退路径。
4. **融合命中**：静态 W8A8 等预期触发 Norm+Quant 融合的场景检查 `match_table`，并验证融合后的数值。
5. **权重布局**：静态 W8A8 至少覆盖 NZ=0/1 及 TP 切分。
6. **KV 精度独立记录**：W8A8C8、W4A8C8 和 FP8 模型分别记录权重精度与 KV Cache 精度。GLM-5.3-Flash-w8a8c8 需对照 vllm-ascend #15925 的已知精度问题做确认。
7. **量化 draft**：MTP/DSpark/DFlash2 开启时，同时记录 target 和 draft 的量化格式。
8. **精度和性能基线**：量化 Inductor 与同量化 eager 比正确性，与同量化生产默认后端比性能；官方教程已发布 AISBench 参考值的模型直接引用。
9. **异常状态单列**：`xfail`、`skip`、缺权重和 `eager blocked` 均不计为通过。

## 6. 正确性、性能和路径证据

### 6.1 正确性

- Greedy 场景优先比较完整 token 序列。
- 对 prefill/decode logprobs 和融合算子设置明确容差。
- MoE near-tie 分歧需要单独进行 logprob、任务精度或高精度参考仲裁，不能直接以 xfail 结案。
- 新主模型至少执行任务精度评估；量化模型必须与同量化 eager 基线比较。
- 采样、MTP/DSpark、异步调度和图回放需要检查 RNG、接受率以及连续请求的一致性。

### 6.2 性能

固定负载至少包括：

- `ISL/OSL=128/512，batch=1`。
- `ISL/OSL=1024/256，batch=1/8/32`。
- `ISL/OSL=4096/128，batch=8`。
- 新模型增加 32K 长上下文；声明支持更长上下文的模型增加对应专项。

性能阈值：默认配置下，端到端吞吐不低于当前生产默认后端的 95%，TPOT 劣化不超过 5%，TTFT 劣化不超过 10%。每组负载至少跑 3 次取中位数并记录方差；方差超过 5% 时追加运行次数确认。记录 TTFT、TPOT、端到端吞吐、冷编译时间、暖启动时间、峰值显存和长时间服务稳定性。

### 6.3 Inductor 路径证据

每个测试项至少保存一类能够证明真实进入 Inductor 的证据：

- 编译、capture 和 replay 计数。
- `output_code.py`、Triton/TE 产物或 `.npubin`。
- 最终 `cudagraph_mode` 和 compiler backend。
- 预期融合的 `match_table`。
- 每个 rank 的独立缓存和日志。

仅生成非空文本不能证明 Inductor 路径正确，也不能替代精度和性能验收。

## 7. CI 分层

| 层级 | 执行测试项 | 内容 |
|---|---|---|
| PR | 1、12 + 12 的 TP2 smoke | 小模型基础生成、全部关键图模式、配置和编译缓存；增加一条双卡 TP2 smoke（对应现有 `tests/e2e/pull_request/two_card/inductor/test_inductor_track_tp2.py`），防 rank 隔离和 TP 切分布局类回归；另跑配置/Pass/量化 UT |
| Nightly | 2–8、13、14 | 新旧 Dense/MoE、主要量化、MTP、多模态、精度和性能 |
| Weekly | 11、15–21 | Flash-Next 新架构、旧架构、旧多模态、Pooling、传统 MoE、MLA、专项量化和 Dense MXFP8 |
| Release | 1–21 | 全量正确性和默认配置性能；9、10、20 增加多机、PD、长序列和稳定性 |

冷编译使用独立空缓存并单独记录；普通回归可以使用隔离后的暖缓存，但不能把命中旧缓存的用例报告为冷启动。

## 8. 明确精简的模型

以下模型不再单独建立 Inductor 测试项：

| 删除项 | 精简原因 |
|---|---|
| Qwen3.5-4B | 与 Qwen3.5-0.8B 同架构，小模型功能矩阵由 0.8B 承担 |
| Qwen3.5-27B、Qwen3.6-27B BF16/W8A8 | 与 Qwen3.8-27B 同为 Dense 混合注意力架构，Dense 主验收由 Qwen3.8-27B 承担，MXFP8 由 Qwen3.6-35B-A3B 承担 |
| Qwen3-8B BF16 | 传统 Dense BF16 已由 Qwen3-0.6B 覆盖，只保留静态 W8A8 |
| Qwen3-8B-FP8 | FP8 block 改由 Qwen3.8-27B-FP8 覆盖 |
| Qwen3-30B-A3B BF16 | 传统 MoE 只保留更有回归价值的 W8A8 版本 |
| Qwen3-Next-80B-A3B | GDN/混合注意力由 Qwen3.5/3.6 覆盖，不再重复大模型回归 |
| DeepSeek-V3.2、DeepSeek-V4-Flash | DSA/SFA 由 DeepSeek-V4.1 和 GLM-5.3-Flash 覆盖；V4-Flash 被 V4.1 替代 |
| GLM-5.3 旗舰版 | 超出本次资源和常规 CI 范围；GLM 主线由 GLM-5.3-Flash 承担 |
| GLM-5.1、GLM-5.2 BF16/W8A8 | 与 GLM-5.3-Flash 基座相近；只保留 GLM-5.2-W4A8C8 量化专项 |
| Qwen3.8-2.4T-A95B | 超出本次资源和常规 CI 范围 |
| 全部 310P 部署形态 | Inductor 不支持 310P，显式排除 |

## 9. Release 通过条件

Release 前必须同时满足：

1. P0 项全部形成有效结果，无未解释的 xfail、skip 或配置错误。
2. P1 项完成计划内回归；未执行项必须明确从本次支持范围排除，不能默认视为通过。
3. 所有通过项都有同权重、同量化、同 KV 精度的 eager 对拍和真实 Inductor 路径证据。
4. 默认配置正确性通过，性能满足第 6.2 节的阈值。
5. 原有 eager、默认 compiler、ACLGraph、量化和分布式测试无回归。
6. 结果单独标记 `passed`、`failed`、`xfail`、`skipped` 和 `eager blocked`，不得汇总成未经区分的"全部通过"。

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
3. eager 仅作为正确性参考，默认不运行性能测试；性能必须与当前生产默认后端进行同日、同环境、同负载比较。需要 eager 性能数据定位问题时，作为专项诊断单独运行，不纳入常规流程。
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
| 21 | `Eco-Tech/Qwen3.8-27B-w8a8-mxfp8` | W8A8-MXFP8 | 新 Dense 的 MXFP8 Linear/kernel 和数值路径；复用官方 AISBench 基线，不重复完整特性矩阵 | P1 |
| 22 | `Kimi-K3-expert16-top8-100B-CountBenchQA` | BF16 | Kimi K3 的 100B 架构验证权重；保留 93 层 KDA+全注意力混合布局和多模态结构，采用 16 专家、top-8 路由；覆盖模型加载、KDA/MLA、MoE 和 Inductor 编译；不能替代完整 896 专家、top-16 K3 的精度、性能及 EP 扩展性验收 | P1 |

## 4. 特性与测试项的固定分工

以下分工直接指定到第 2、3 节编号，不再额外做模型和特性的自由组合。正确性、性能和编译缓存的通用规则见第 6 章，不在本表重复展开。

| 特性 | 指定测试项 | 覆盖要求 |
|---|---|---|
| 图模式 | 1–22 | 1、12 显式覆盖 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`，只做功能、正确性和编译/捕获/回放证据检查；其他项验证默认 O2 下实际生效的图模式，高风险路径补 `PIECEWISE` 或 `FULL_DECODE_ONLY`；不逐一运行完整性能负载 |
| 优化级别 | 1–22 | 当前均使用默认 O2，其他优化级别本轮暂不覆盖；优化级别与图模式分别记录，不以显式图模式结果代替优化级别覆盖 |
| 执行层 | 1–22 | 以 MRV2 为主；仅 12、14、16、19 增加 MRV1 兼容回归，分别覆盖传统 Dense、静态量化、MLA 和 Pooling。MRV2 不支持或失败时记录真实状态，不回退 MRV1 冒充通过 |
| 多模态 | 1、2、3、4、18、22 | 纯文本、单图、多图、不同分辨率、图文混合 batch；2–4 增加视频；18 验证传统 encoder；22 验证 K3 多模态和 `use_rot_proj` 路径 |
| MTP/DFlash2 | 1–8、11 | 按模型实际提供的 draft 权重验证；同时检查 target 和量化 draft 的组合 |
| DSpark | 9、10、20 | 检查输出、接受率、图回放和 prefix cache 组合 |
| ngram | 12 | 保留通用非模型 draft 的推测解码回归 |
| Prefix Cache、Chunked Prefill | 1–17、22 | 12 增加传统小模型基础回归；新模型重点检查混合 KV、量化 KV、MTP 和图回放组合；22 重点检查 KDA/全注意力混合 KV |
| LoRA | 1、5、12 | 分别覆盖新小模型、新 MoE 和传统 Dense；包含单 LoRA、Multi-LoRA 和运行时加载 |
| LoRA × 量化 | 14 | 静态 W8A8 模型挂 LoRA，验证量化 kernel 与 LoRA 分支同时生效且数值正确 |
| 异步调度 | 2、5、9 | 默认开启路径下验证图回放、采样 RNG、连续请求一致性；与 prefix cache、MTP 组合 |
| 编译失败与回退 | 1、12 | 构造含不支持算子或强制 graph break 的配置，验证按预期回退且日志明确记录，不静默改变精度模式 |
| 结构化输出 | 1、12 | 约束采样与编译、图回放组合的输出合法性 |
| TP/EP/EPLB | 5–11、15、22 | Qwen MoE 覆盖 TP2/TP4 和 EPLB；DeepSeek/GLM 按推荐拓扑覆盖 TP/EP；22 验证 16 专家拓扑，不外推完整 896 专家路径 |
| PCP/DCP/PD | 9、10、20 | 验证真实 KV 传输、D 侧不重算 prefill、长序列和异常恢复 |
| 长序列 | 2–11、20、22 | 常规 32K；按模型支持边界执行 128K/256K；1M 仅作为明确支持模型（9、10）的专项 |
| Pooling/Embedding | 19 | 不同输入长度和 batch，对比 eager 的向量或分数 |
| 在线服务 | 1、2、5、9、10 | OpenAI 接口、连续请求、并发、thinking、tool calling 和服务重启 |

DeepSeek-V4.1、Kimi K3 等可能受 Breakable CUDAGraph 影响的模型，必须确认实际编译未被打断。需要显式设置 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 的场景，应在测试配置和结果中记录。

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

## 6. 编译缓存、正确性、性能和路径证据

### 6.1 编译缓存

冷编译使用独立空缓存并单独记录；普通回归可以使用隔离后的暖缓存，但不能把命中旧缓存的用例报告为冷启动。

相同权重、源码和运行配置的后续正确性测试、性能测试及失败重跑应复用对应缓存目录，避免重复冷编译。脚本只负责指定缓存目录并记录运行前状态，缓存命中仍由 vLLM、Inductor 和 Triton 自行判断。每次运行保存实际缓存路径、框架生成的缓存标识及 `cold/warm` 状态：`cold` 表示运行前为空目录，`warm` 表示复用已有目录；`warm` 本身不等于所有产物都命中缓存。

权重、源码、后端、图模式、TP 拓扑或其他编译配置变化，以及缓存损坏或无法确认隔离关系时，必须使用新的空缓存。Release 至少保留一次受控冷启动验收，不因日常暖缓存回归而省略。

### 6.2 正确性

#### 6.2.1 基本要求

- 正确性测试与性能测试使用独立入口、独立进程和独立结果目录，不在同一任务中连续执行。两类测试不共享运行状态，正确性结果不自动阻断独立的性能任务。
- 正确性检查以同权重、同量化、同 KV 精度的 eager 为参考。Greedy 生成时优先比较完整 token 序列，同时保存解码后的文本。
- 对 prefill/decode logprobs 和融合算子，根据数据类型与量化格式设置明确容差，不能只凭非空生成判断通过。
- MoE 候选分数接近而导致输出分歧时，先记录并比较 logprob；必要时结合标准任务精度或更高精度参考进一步判断，不直接以 xfail 结案。
- 新主模型至少完成一项标准任务精度评估；量化模型必须与同量化 eager 基线比较，并报告分数差。
- 对采样、MTP/DSpark、异步调度和图回放，检查随机种子、接受率、连续请求一致性以及实际图回放证据。
- 输出检查、logprob 和 AISBench 任务可在不同空闲 NPU 上并行执行；每个任务使用独立缓存、端口和结果目录。

#### 6.2.2 固定输入与 logprob 检查

固定输出检查使用 4 组短输入和 4 组长输入，均为固定自然语言内容，不使用随机 token ID 代替正确性输入。每组生成 2 轮，`max_tokens=1024`、`ignore_eos=False`、`logprobs=5`、`temperature=0`。逐请求记录实际输入/输出 token 数、文本、token 序列、top-5 logprobs、两条路径的差异及重复运行稳定性。输出预算允许正常 EOS 结束，不要求生成满 1024 tokens。

8 组固定输入如下：

| 编号 | 具体内容 |
|---:|---|
| 1 | `The capital of France is` |
| 2 | `请比较光合作用与细胞呼吸的能量转换过程，并说明二者之间的联系。` |
| 3 | `请解方程组：3x + 2y = 16，x - y = 2。写出完整计算过程。` |
| 4 | `Write a Python function that reverses a string.` |
| 5 | 医学与公共卫生文档，介绍疫苗、免疫反应、群体保护和接种风险；固定为 8 个部分，最后要求用三句话说明疫苗的作用及接种注意事项 |
| 6 | 法律与个人信息保护文档，介绍合法处理、知情同意、最小必要、数据安全和个人权利；固定为 8 个部分，最后要求用两句话说明处理个人信息时必须遵守的核心原则 |
| 7 | Climate science document covering the greenhouse effect, observed warming, human influence, mitigation, and adaptation; fixed as 8 sections, ending with a request to explain the main causes of climate change and the corresponding responses in three sentences |
| 8 | History document covering the printing press, book production, literacy, universities, religious change, and the spread of scientific knowledge; fixed as 8 sections, ending with a request to explain how printing changed European society in three sentences |

长输入由对应主题的固定自然语言段落组成，记录实际 token 数，不补齐到人为指定的相同长度。

logprob 比较用于分析数值差异，不能替代正常生成、decode 图回放和任务精度。发生 token 分叉后，不直接比较分叉之后因上下文不同而产生的 logprob；需要固定上下文时，使用相同 prompt+answer 的 teacher-forced logprob 比较。

#### 6.2.3 标准任务精度

标准任务精度固定使用以下分工，日常小子集必须固定样本 ID、顺序和 seed，Release 使用完整任务集或官方推荐配置：

| 模型范围 | 题目集 | 执行规则 |
|---|---|---|
| 文本生成模型 | MMLU、GSM8K | Nightly 各固定 200 题；Release 分别使用 MMLU 全量和 GSM8K 全量 1319 题 |
| Qwen3.8-27B、Qwen3.6-35B-A3B 及量化版本 | GPQA Diamond | Weekly/Release 使用全量 198 题 |
| Qwen2.5-VL-7B-Instruct | TextVQA | Nightly 固定 200 张图片；Release 使用官方验证集 |
| Qwen3-Embedding-0.6B | STS-B | 使用全量 1379 对句子，报告 Spearman 相关系数 |

MMLU、GSM8K、GPQA 和 TextVQA 统一使用 AISBench 官方任务配置；STS-B 在独立 Embedding 脚本中计算。每条路径分别记录任务分数、样本数、预测结果、评分结果和配置，报告 Inductor 相对同权重、同量化 eager 的百分点差值。

所有文本生成模型的 MMLU、GSM8K 单题输出上限统一固定为 2048 tokens，不按模型区分，用于限制异常持续生成造成的耗时；输入上下文不做相同截断。结果配置中记录该上限，达到上限的样本仍按任务原始评分规则处理。

#### 6.2.4 本轮不扩展的任务

本轮不在常规矩阵中扩展以下任务，原因不是这些任务无效，而是当前模型范围和测试成本下优先级较低：

| 任务 | 本轮不扩展的原因 |
|---|---|
| C-Eval | 与已选择的 MMLU 都覆盖综合知识和多项选择能力；本轮需要控制重复任务数量，先固定使用 MMLU |
| MATH-500 | 与 GSM8K、GPQA 已覆盖的数学推理存在重叠，且输出更长；本轮优先保留一个基础数学任务和一个高难度推理任务 |
| AIME 2024 | 只有 30 题，1 题差异就约为 3.33 个百分点，小样本波动过大，不适合作为本轮常规回归门禁；仅在明确的高难度推理专项中使用 |
| LiveCodeBench | 当前矩阵没有独立代码模型，长代码生成会显著增加执行时间；本轮仅通过固定代码提示词覆盖基础代码生成路径，后续加入代码模型时再建立专项 |

### 6.3 性能

性能测试只比较当前生产默认后端与 Inductor。eager 仅提供正确性参考，不在常规流程中执行预热、吞吐或延迟计时；确有性能诊断需求时单列为专项，不与生产基准混合。

性能测试不以 token 一致、文本完全一致、重复生成稳定或全部输出正常为前提。只要服务能够启动、请求能够完成并产生输出，即可独立执行性能测试；token 差异、logprob 差异、部分空白或异常文本均保留在正确性结果中，不阻断性能测量。模型加载失败、引擎报错或请求无法完成时，记录运行失败，不生成无效的性能数据。

非默认图模式只做功能和正确性检查，不逐一运行完整性能负载。完整性能测试使用生产默认配置，并记录默认 O2 预设最终选择的图模式。某个非默认图模式准备投入生产时，再单独增加对应性能测试。优化级别不作为图模式矩阵的隐式扩展项。

性能负载统一使用官方 `vllm bench serve`/`vllm bench` 工具生成和统计，不在测试框架中重复实现请求计时和指标计算。default 与 Inductor 必须使用相同的数据文件、样本顺序、请求数、QPS、并发和生成参数。

固定负载分为以下三组：

| 负载 | 适用范围 | 配置 |
|---|---|---|
| ShareGPT 在线服务 | 所有文本生成模型 | 使用 `ShareGPT_V3_unfiltered_cleaned_split.json`；每个 QPS 运行 200 条请求；QPS 固定为 1、4、16、inf；预热 10 条请求 |
| 长输入合成负载 | 新模型及声明长上下文的模型 | 使用 `vllm bench serve --dataset-name random`；8192/1024 配置 32 条请求，32768/1024 配置 16 条请求；并发 8，请求速率 inf，`random_range_ratio=0`、`random_prefix_len=0`、`ignore_eos=True`、seed 0 |
| 多模态与 Pooling | 对应模型 | 多模态沿用官方 VisionArena 配置；Pooling/Embedding 使用官方对应后端和负载，不用文本生成负载替代 |

ShareGPT 用于模拟真实对话请求，不能代替明确长度的长上下文测试。随机 token ID 只用于合成性能负载，不用于正确性输入。所有负载保存官方原始 JSON、逐请求结果、TTFT、TPOT、ITL、端到端吞吐及 p50/p95/p99 分位数。

性能阈值：默认配置下，端到端吞吐不低于当前生产默认后端的 95%，TPOT 劣化不超过 5%，TTFT 劣化不超过 10%。常规测试与修复验证每组负载运行 1 次；结果异常或接近阈值时追加测量。需要多次测量时，按 default → Inductor 的顺序交错执行，不能先完成一条路径的全部重复再运行另一条路径。记录 TTFT、TPOT、端到端吞吐、冷编译时间、暖启动时间、峰值显存和长时间服务稳定性。

性能测试必须独占所使用的 NPU。启动前检查进程和 HBM 占用，运行期间不调度其他任务，并使用卡级锁避免并发任务绑定同一 NPU。


### 6.4 Inductor 路径证据

每个测试项至少保存一类能够证明真实进入 Inductor 的证据：

- 编译、capture 和 replay 计数。
- `output_code.py`、Triton/TE 产物或 `.npubin`。
- 最终 `cudagraph_mode` 和 compiler backend。
- 预期融合的 `match_table`。
- 每个 rank 的独立缓存和日志。

仅生成非空文本不能证明 Inductor 路径正确，也不能替代精度和性能验收。

## 7. CI 分层

| 层级 | 执行测试项 | 正确性与任务精度 | 性能、缓存与专项 |
|---|---|---|---|
| PR | 1、12，另加 12 的 TP2 smoke | 8 组固定输入、2 轮生成、1024 输出预算、`logprobs=5`；5 个显式图模式的功能与正确性；默认 O2；TP2 的 rank 隔离和 eager 输出比较；配置、Pass、量化 UT。不运行标准任务精度 | 不运行 ShareGPT 和长上下文性能负载；只保留编译、图捕获/回放及配置证据 |
| Nightly | 1–8、13、14 | 每个模型执行固定输出检查和 logprob 比较；MMLU 固定 200 题、GSM8K 固定 200 题；使用默认图模式比较 eager 与 Inductor | ShareGPT 200 请求，QPS 1、4、16、inf；保存 TTFT、TPOT、ITL、吞吐和分位数。模型 2、3、5–8 增加 8192/1024（32 请求）和 32768/1024（16 请求），并发均为 8。每组负载运行 1 次，使用暖缓存并记录实际状态 |
| Weekly | 9、10、11、15–22 | 按第 4 章分工执行架构专项；模型 2、3、5–8、21 的代表性路径运行 GPQA Diamond 全量 198 题；Qwen2.5-VL 运行 TextVQA 固定 200 张图片；Qwen3-Embedding 运行 STS-B 全量 1379 对句子 | 长上下文、MTP/DSpark、LoRA、EPLB、PCP/DCP/PD 按第 4 章分工执行；验证冷启动、暖启动、rank 隔离、配置隔离和损坏缓存恢复；多机及 PD 仅在资源和权重满足时执行 |
| Release | 1–22 | 所有计划内模型完成正确性检查；MMLU 全量、GSM8K 全量 1319 题；指定大模型完成 GPQA 全量 198 题；TextVQA 使用官方验证集，STS-B 使用全量；保存预测、评分和配置 | 完成 ShareGPT 全部 QPS 和长上下文负载；每组性能负载按 default → Inductor 交错运行 3 轮并记录中位数和波动；执行一次受控冷启动、暖启动和缓存隔离；记录峰值显存、长时间服务稳定性及异常恢复；执行原有 eager、default compiler、ACLGraph、量化和分布式回归 |

分层原则：

1. P0/P1 表示重要程度，不表示执行频率。
2. PR 重基础回归，Nightly 重主要模型和生产路径，Weekly 重架构专项，Release 重完整验收。
3. 同一模型需要重复执行前层内容时，可以复用历史证据判断是否需要重跑，但不能把历史结果冒充本轮结果。
4. 每层都独立记录 `passed`、`failed`、`xfail`、`skipped` 和 `eager blocked`。缺权重、资源不足及 eager 失败均不计为通过。
5. 后续层级的新结果不能覆盖或改写此前层级的失败记录。

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

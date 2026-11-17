调研下当前ascend pytorch的当前的triton experimental（inductor）开发情况，有哪些特性功能已支持？目前本地工作区vllm-ascend 的indcutor接入支持情况如何？

# 一、Ascend/pytorch 的 triton_experimental 开发现状

## 1. 开发集中在官方 master 分支（对应 torch 2.15）

- **架构**：`triton_experimental` 已合入 torch_npu 官方仓库，是正式注册的内置编译后端（位于 `pytorch/torch_npu/_inductor/triton_experimental/`，首个官方提交在 2026-07-23）。通过环境变量 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental` 或 `torch.compile(options={"npu_backend": "triton_experimental"})` 启用，取代了早期外挂式的 monkey-patch 补丁包 `npu_inductor`。
- **分支差异**：官方 master 上该后端累计已有 **78 个提交**且仍在活跃开发（本地克隆的 `origin/master` 上最新提交是 2026-11-16 的 fast launcher 修复）；而 **v2.13.0 和 v2.14.0 两条发布分支的后端代码停在 9 月 21 日**，master 上的新改动没有同步过去（详见 `doc/AI文档/02-上游生态现状.md`）。
- **master 分支 10–11 月新增的能力**（本地使用的 2.13 分支没有）：FlexAttention 适配（10-13）、NPU GEMM 模板与调优启发式（10-12）、ComboKernel 多核合并（10-10）、静态启动器（9-30）、基于 vectorcall 的快速启动器（11-05）、实验性 C++ wrapper（11-09）、Dynamo 预编译产物支持（11-12）、坐标下降调优（11-07），以及一批"硬件解耦"方向的正确性修复（cumsum、KLDiv、BCEWithLogits 的分解规则、按数据类型区分的精度提升等）。

## 2. 本地 2.13 版本已支持的功能（按代码盘点）

代码主体约 6100 行 Python，其中自动调优与启动器（`npu_triton_heuristics.py`）占 2991 行。功能面如下：

| 层面        | 已支持的能力                                                 |
| :---------- | :----------------------------------------------------------- |
| 代码生成    | 面向 NPU 的扁平化索引生成；reduction 树的真实分块提升与自动切块；跨核的 reduction 轴拆分（40 核各算一部分再合并）；网格数超过 65535 时自动折叠；多核分组调度；标量化的小维度循环；int64 到 int32 的边界类型下推与去重；min/max reduction 的 NaN 语义保持；int32 索引溢出保护 |
| 自动调优    | 一维逐元素算子和 reduction 的公式化配置生成（另有级联列表作为备选）；增强版配置扫描；支持 mspti 设备计时和 NPU event 计时两种实测方式；调优结果缓存以 torch_npu 版本号加后端源码摘要作为键 |
| 算子覆盖    | 重写了 81 个 aten 算子的 lowering（包括 mm/bmm/addmm、卷积及其反向、var_mean、glu、二维 padding、双线性上采样等）；gather 类算子（index_select、slice+cat、argmax）已有原生实现，可融合成单个 kernel |
| 性能专项    | permute+gather 的三路分派（寄存器 gather / 转置 / 物化，按步长和 UB 容量选择）；clone 折叠策略（判断折叠还是物化，并有第二道防线校验）；softmax/log_softmax 按宽度选择走 Triton 融合还是 aclnn；Ascend950 专属的 embedding 走 CANN 寄存器级 gather；带 mask 的 `_safe_softmax` 走 aclnn |
| 图优化 pass | 消除 int→float→int 的往返转换、折叠 Max(1,·) 表达式、禁用 addmm 融合和 shape padding 等 |
| 配置体系    | `config.py` 集中管理所有配置项（带类型、支持 `config.patch` 临时覆盖、参与缓存 hash 计算），不依赖环境变量 |



**明确不支持的部分**（历史审计结论）：AOT 编译 / Standalone / MEGA（C++ wrapper 代码生成器未注册，是遗留清单里唯一的 P0）、图内随机数、cond/while_loop/flex_attention 等高阶算子、torchao/GPTQ/AWQ 的量化 GEMM lowering、集合通信的编译支持、fp64 和复数等长尾算子。其中一部分（flex_attention、GEMM 模板、C++ wrapper）master 分支在 10–11 月已经补上，本地 2.13 分支尚未合入。

# 二、本地 vllm-ascend 的 Inductor 接入情况

## 1. 代码状态

- 分支 `hanya/feature/inductor_backend-1109`，HEAD 为 `81f53624f`（2026-11-12），基于 **vllm-ascend 0.27.1rc + vllm 0.27.1**，已完成对 torch 2.13、C++20、CANN 9.2 的适配（包括 11-12 修复 Ascend950 上 MLA prolog 日志宏导致的构建失败）。
- **启用方式**：用户设置 `compilation_config.backend="inductor"`（这是 vLLM 官方的编译后端配置项），vllm-ascend 平台代码随后自动设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`（见 `vllm_ascend/platform.py:505`），编译请求经由 vLLM 的 `compile_fx` 进入该后端，不需要额外安装补丁包。这与 vllm-ascend 官方路线不同——官方明确不用 Inductor（`use_inductor=False`），社区的接入提案 RFC #8010 至今无人跟进，本地这套是独立实现并经过系统验证的。
- **配置处理**：`platform.py` 在启用该后端时会做一系列配置调整——用户未显式设置时把 `custom_ops` 设为 `["all"]`（9 月底的性能期裁定）、强制关闭几个不支持的图优化选项和 combo_kernels、校验 `inductor_compile_config` 里的配置键是否合法（可用 `VLLM_ASCEND_STRICT_INDUCTOR_CONFIG=1` 改成严格报错）、图捕获模式跟随 `-O` 级别派生（O2/O3 默认 FULL_AND_PIECEWISE）。
- **图捕获**：ACLGraph 支持 NONE（不捕获）/ PIECEWISE（分段）/ FULL（整图）/ FULL_DECODE_ONLY / FULL_AND_PIECEWISE 全部档位；图融合 pass 包括 Norm+Quant、QKNorm+RoPE、Muls+Add 三个（SwiGlu+静态 W8A8 量化的融合 pass 已在 `fe4fc22f0` 中撤销）。
- **测试**：单元测试 290 多个用例（编译配置、pass 等）；端到端测试分三档——单卡 9 个文件（分段捕获、整图模式、W8A8、MLA、长序列 16K、投机解码、FP8 等）、双卡 7 个（TP2、32B 模型、MLA、PCP、MXFP8）、四卡 3 个（MoE TP4、MoE W8A8、DeepSeek-V4 W4A8）。

## 2. 实测结论

- **九模型性能基线**（2026-09-26，910 系列机器）：Inductor 编译路径对比 vllm-ascend 原有默认编译路径，整体吞吐比值 **0.93–1.11**；Qwen3-Next-80B（混合线性注意力）收益最大（批量 32 时快 26%），DeepSeek-V2-Lite BF16 的 MLA 解码是唯一明显回退（慢约 7%）。W8A8 量化已追平略超（1.01），投机解码（eagle3/dflash/ngram/MTP）输出与 eager 逐 token 一致，编译缓存跨进程复用带来 8–11 倍启动提速。
- **A5（Ascend950DT）最新实测**（`server-scripts/bench213/`）：Qwen3-0.6B 冒烟通过（首次冷编译 406.7 秒）；11-16 对 Qwen3.5 系列修复后复测——0.8B / 2B / 35B-A3B 两条编译路径都跑通，12 条提示词中逐 token 一致 9、10、8 条（不一致处是语义合理的分叉，不是乱码），性能比值约 **0.99–1.05**；35B-W8A8-MTP 两条路径都以退出码 143 终止（原因待查）；11-17 正在做 DeepSeek-V4-Flash-FP8 的 eager 验证（目的是确认 `KvQuantSparseAttnSharedkv` 的 tiling 错误是否只在图编译路径出现）。
- **正在处理的 bug**：Qwen3.5 的 GDN 后置 LayerNorm 问题已定位——triton_experimental 把用户自写的 Triton kernel 节点错误地送进了通用兜底逻辑，加上 `eps` 参数被错误推导为 FP64。本地临时修复方案已在 A5 上端到端验证通过，但**还没有合入任何正式分支**——本地 pytorch 检出的 `hanya/bugfix/layernorm_1117` 分支上唯一的新提交 `54fc6540b9`（11-17，保留用户自定义 Triton kernel）就是这个修复在 torch_npu 侧的部分。

## 3. 主要遗留工作（目标 2026 年 12 月底合入官方仓库）

按 `doc/AI文档/feature/Inductor适配功能性能与覆盖缺口.md` 的整理：最高优先级是 **vllm-ascend 从 0.27.1 迁移到 0.30**、**统一依赖版本声明**（仓库声明里仍写 triton-ascend 3.2.2，而 A5 实际用的是 3.6.0.post2）、**把 Inductor 用例接入 A5 的 PR CI**（目前 A5 专用测试列表里没有 Inductor 用例）。次一级包括：AOT 编译支持、MLA BF16 解码 7% 的回退、gather 类 kernel 的质量、LoRA/EPLB/PD 分离/多模态的场景验证。另外测试代码本身有一些已知问题：部分 W8A8 测试用例实际没有传 `backend="inductor"`（测试名与真实路径不符）、MoE W8A8 TP4 用例里写了一个配置类不接受的字段等（详见 `doc/AI文档/vllm-ascend-Inductor适配现状.md` 第四节）。

**一句话总结**：triton_experimental 在官方 master 上活跃演进（10–11 月密集补齐了 FlexAttention、GEMM 模板、启动器等），本地落在 2.13 发布分支加一个在途修复；vllm-ascend 侧的 Inductor 接入已是一套完整实现（官方配置项入口、配置自动处理、全部图捕获档位、图融合 pass、分层测试），910 系列上九个模型性能整体持平偏好（0.93–1.11 倍），当前工作重心是 A5 机器上 Qwen3.5 和 DeepSeek-V4 的问题修复，以及 12 月底合入官方前的版本迁移（0.30）和 CI 建设。
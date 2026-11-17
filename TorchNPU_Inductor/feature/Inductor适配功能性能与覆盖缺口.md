# vllm-ascend Inductor 功能缺失、性能优化与覆盖缺口

## 1. 范围和使用方式

本文汇总 Inductor 适配中不属于单纯 bug 的遗留事项，分为：

- **功能缺失**：上游已有能力，但 torch_npu、triton_experimental、triton-ascend 或 vllm-ascend 尚未提供。
- **性能优化**：功能可运行，但 kernel、图模式、编译耗时或默认策略仍未达到目标。
- **覆盖缺口**：模型、量化、分布式、调试和 CI 组合尚未得到可靠验证。
- **工程缺口**：版本迁移、依赖矩阵、社区上游化和交付物尚未收口。

崩溃、数值错误、挂死和测试自身错误见配套文档：[Inductor 适配 Bug 定位与修复追踪](../bugfix/Inductor适配Bug定位与修复追踪.md)。两文有交叉时，以 bug 文档跟踪缺陷，以本文跟踪其带来的功能或验证影响，避免重复立项。

核对基线：

- vllm-ascend：`hanya/feature/inductor_backend-1109@81f53624f`。
- 历史问题主要来自功能遗留总档、Stage 5 TODO、Stage 6 结果和当前环境/升级文档。
- **本次只做源码和文档静态整理，没有重新运行 NPU 测试。**“open”表示历史总档未闭环；涉及版本升级的项目，必须在目标 A5/CANN/triton-ascend 组合上复测。

链接说明：每行“来源”都可点击。`PI-*`、`TODO-*` 是源文档内的唯一定位编号，跳转后可用该编号搜索；代码来源在链接文字中附行号。`#Lxx` 是代码行定位片段，具体编辑器是否直接跳到行取决于其本地文件链接支持，但行号本身始终可用于定位。

## 2. 当前优先级概览

| 优先级 | 当前事项 | 类型 | 来源（定位） |
|---|---|---|---|
| P0 | 从 vLLM/vllm-ascend 0.27.1 迁移到目标版本，并合并本地 Inductor 改动 | 工程/版本 | [升级评估：风险与执行顺序](../04-升级评估.md) |
| P0 | 统一 CANN 9.2、triton-ascend 3.6 与仓库仍 pin 3.2.2 的支持矩阵 | 工程/依赖 | [pyproject.toml:21](<../../../inductor-meta-worktree/vllm-ascend/pyproject.toml#L21>)、[当前情况](../../手工文档/当前情况.md) |
| P0 | 修复测试配置并接入真实 A5 CI，再评估功能完成度 | 覆盖/CI | [适配现状：测试与 CI](../vllm-ascend-Inductor适配现状.md) |
| P1 | AOT/Standalone/MEGA、`cpp_wrapper` 等序列化编译产物路径 | 功能缺失 | [PI-TN-01/02](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| P1 | DeepSeek-V2-Lite BF16 decode 历史回退约 7%，以及 gather/GEMM kernel 质量 | 性能优化 | [Stage6 头条表](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| P1 | LoRA、量化、EPLB、PD、多模态等组合覆盖 | 覆盖缺口 | [PI-VA-03/04/05/12](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| P1 | 默认配置无回退得到证明后，再决定是否将 Inductor 翻成默认轨 | 性能/默认策略 | [Stage5 TODO-VA-1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>)、[platform.py:219](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L219>) |

## 3. 功能缺失与上游能力对齐

### 3.1 torch_npu / triton_experimental / triton-ascend

| 编号 | 类型 | 功能缺口 | 当前状态和影响 | 来源（定位） |
|---|---|---|---|---|
| PI-TN-01 | 功能缺失/P0 | NPU `cpp_wrapper` C++ 代码生成器未注册 | 历史总档为唯一 P0；阻塞完整 AOT wrapper 路径，需先补 TE 能力 | [torch 章 PI-TN-01](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-02 | 功能缺失/P1 | AOT、Standalone、MEGA-AOT 编译产物路径未适配 | Standalone/MEGA 当前 raise；AOT=1 可运行但跨重启复用未证。普通 FxGraphCache 暖启动已通，不应混为一谈 | [torch 章 PI-TN-02](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)、[platform.py:507](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L507>) |
| PI-TN-03 | 功能缺失/P1 | `triton_kernels`/Triton 3.5 API 断链 | 旧 3.2.2 下影响 gpt_oss/mxfp4 等模型；当前 3.6 是否补齐需要重新验证 | [torch 章 PI-TN-03](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)、[Stage6 #2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| PI-TN-04 | 功能缺失/P1 | graph-safe RNG | RNG 落 extern 或捕获后 seed/offset 回放语义不等价；当前仅有图内 bernoulli 警告 | [torch 章 PI-TN-04](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)、[PassManager:125](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/ascend_post_grad_pass_manager.py#L125>) |
| PI-TN-05 | 功能缺失/P1 | cond、while_loop、invoke_subgraph、flex_attention 等高阶算子 | 历史审计预期直接编译失败；只有相关模型触发时才进入当前支持范围 | [torch 章 PI-TN-05](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-06 | 功能缺失/P1 | torchao/GPTQ/AWQ/int8 weight-only 等量化 GEMM lowering | 与 vllm-ascend 已有 ModelSlim/W8A8 路径不是同一支持面，不能由 W8A8 通过推导全部量化可用 | [torch 章 PI-TN-06](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-08 | 功能缺失/P1 | `allow_buffer_reuse` 被跨后端 import 副作用关闭 | 可能影响 buffer 复用和显存；需消除隐式全局副作用并验证缓存一致性 | [torch 章 PI-TN-08](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-12 | 功能缺失/P1 | 多个 FX Pass 的 device 守卫只识别 CUDA/XPU | NPU 上部分用户开关静默无效；需逐文件核对，而不是统一强开 | [torch 章 PI-TN-12](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-19 | 功能缺失/P1 | 多个合法 Inductor 选项被 TE/VA 禁用 | `split_reductions`、shape padding、layout、coordesc、combo 等尚无适配；VA 当前通过告警、覆 False 或 raise 规避 | [torch 章 PI-TN-19](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)、[platform.py:1395](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L1395>) |
| PI-TN-10 | 功能缺失/P2 | `size_asserts` 被全局关闭 | 影响调试期尺寸断言，不是推理主链阻塞项 | [torch 章 PI-TN-10](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-11 | 功能缺失/P2 | fp64、复数、部分 SDPA/RNN 长尾算子 | 受 aclnn 算子供给限制，fallback 也无法保证等价覆盖 | [torch 章 PI-TN-11](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-13 | 功能缺失/P2 | 集合通信编译 lowering 和 micro-pipeline | 影响通信与计算融合、TP 优化；不能用 eager HCCL 可用代替编译面支持 | [torch 章 PI-TN-13](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-14 | 功能缺失/P2 | AsyncCompile 子进程池、远程共享缓存、bisector、mode 预设等基础设施 | 影响冷编、分布式缓存分发和调试效率 | [torch 章 PI-TN-14](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-TN-20 | 功能缺失/P2 | unbacked SymInt/varlen 全链 | slice/index 可能 extern 切断融合，tile 还有固定回退；主流模型未触发不等于能力已具备 | [torch 章 PI-TN-20](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |

### 3.2 vllm-ascend 平台与维护性

| 编号 | 类型 | 功能/工程缺口 | 当前状态和影响 | 来源（定位） |
|---|---|---|---|---|
| PI-VA-06 | 功能缺失/P1 | FP8 per-tensor/per-channel 原生 checkpoint | 与 FP8 block 的 eager 崩溃不同，这里是平台明确不支持；不能混为同一问题 | [VA 章 PI-VA-06](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-07 | 功能缺失/P2 | STOCK_TORCH_COMPILE mode 1/2 对照路径 | 缺少上游 stock `torch.compile` 对照，影响问题定界，不等于 InductorAdaptor 主轨不可用 | [VA 章 PI-VA-07](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-09 | 功能缺失/P2 | 投机解码 custom_class 兜底和 gemma4_mtp 分发 | 属于平台既有分发缺口；Eagle3/DFlash/ngram 已验证不能覆盖这些分支 | [VA 章 PI-VA-09](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-10 | 功能缺失/P2 | 量化算法和权重格式支持面有限 | GPTQ/AWQ 等算法性方案与 ModelSlim 格式不互通，需要按实际 checkpoint 建支持矩阵 | [VA 章 PI-VA-10](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-29 | 功能缺失/P2 | 平台未自动强制 spawn | fork 可能触发 torch_npu 子进程重初始化错误，当前仍主要依赖部署脚本手动设置 | [VA 章 PI-VA-29](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-15 | 优化重构/P2 | 融合 Pass 注册列表多处手工同步 | 容易造成轨间静默分叉，需要机械一致性检查或统一注册表 | [VA 章 PI-VA-15](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-16 | 优化重构/P2 | Pass UUID、缓存因子和观测依赖手工维护 | 漏加配置或源码因子可能产生脏缓存；与后端 hash 线索联动处理 | [VA 章 PI-VA-16](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[缓存 hash 追踪](../bugfix/Inductor适配Bug定位与修复追踪.md#u-01不同后端复用编译缓存-hash) |
| PI-VA-22 | 优化重构/P2 | 不支持配置的告警/报错等级不一致 | 某些显式无效配置只告警，可能把失败推迟到下一次启动或使用阶段 | [VA 章 PI-VA-22](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[platform.py:1434](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L1434>) |
| TODO-VA-2 | 功能重估 | `use_inductor_graph_partition` 仍被强制 False | 需要验证 TE 支持度和是否能减少 FULL_AND_PIECEWISE 双份编译，不能直接开启 | [Stage5 TODO-VA-2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>)、[platform.py:1153](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L1153>) |
| TODO-VA-3 | 功能缺失/显存 | NPUWorker 未接入完整图捕获显存预估 | 高显存利用率下仍可能在 capture 后 OOM，需要保守预估或专用扣减 | [Stage5 TODO-VA-3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| TODO-VA-4 | 功能缺失/通信融合 | AllReduce+RMSNorm 融合 | TP>1 仍有通信和归一化往返，缺 HCCL 融合核，开发成本较高 | [Stage5 TODO-VA-4](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| TODO-VA-5 | 工程/跟版 | `updatable_graph` 通用化和 reduced capture sizes 清理 | 属上游漂移和版本迁移事项，应随 0.30/0.33 合并处理 | [Stage5 TODO-VA-5](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |

## 4. 性能优化与默认策略

### 4.1 尚未闭环的性能项

| 编号 | 类型 | 性能问题 | 当前证据和下一步 | 来源（定位） |
|---|---|---|---|---|
| PI-TN-21 | 性能/P1 | max_autotune 语义、persistent reduction、GEMM 模板、compile_sizes | NPU GEMM 多走 extern aclnn，盲目开启 autotune 只增加编译时间；需要先补有效模板和推荐档位 | [torch 章 PI-TN-21](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| TODO-TN-1 | 性能/P1 | NPU GEMM Triton 模板 | 与 PI-TN-21 的 GEMM 子项同源；定位是 kernel 提升轴，不是简单打开 default 轨开关 | [Stage5 TODO-TN-1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| PI-TN-22 | 性能/P1 | TE gather 族 kernel 慢于 eager | 历史微基准慢 1.26～1.68×；已证明不是 fallback 切融合，而是生成核离散访存/调度质量 | [torch 章 PI-TN-22](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| STAGE6-MLA | 性能/P1 | DeepSeek-V2-Lite BF16 decode 回退 | Stage 6 同日对拍为 0.93×；需 msprof 定位 TE MLA decode kernel，A5 尚无同口径结果 | [Stage6 头条表](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| PI-VA-19 | 性能/P2 | Inductor QKNorm+RoPE 真图 0 命中 | 注册时序已修，真图 pattern 差集未修；详细缺陷历史在 bug 文档 W-04 | [VA 章 PI-VA-19](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[bug W-04](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| PI-VA-21 | 性能/P2 | legacy QKNorm 0 命中 | 与 Inductor 的 decomposition 原因不同；属于 legacy 优化，不应阻塞 Inductor 主链验收 | [VA 章 PI-VA-21](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-30 | 性能/P2 | TP AllReduce 产物形态与融合收益 | 需要确认产物层 0 命中代表无融合机会还是 pattern 未覆盖，再决定是否投入 | [VA 章 PI-VA-30](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| TODO-TN-3 | 性能/低优 | coordinate descent tuning | 历史宽扫描无收益且冷编约翻倍，需保留能力记录，不建议优先开启 | [Stage5 TODO-TN-3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| TODO-VA-1 | 性能/默认策略 | 是否将 Inductor 翻成平台默认后端 | 当前默认仍是 AscendCompiler；必须先满足全模型正确性和默认性能不劣化 | [Stage5 TODO-VA-1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>)、[platform.py:219](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L219>) |
| COLD-COMPILE | 性能/启动 | 冷编译耗时和 FULL_AND_PIECEWISE 双份编译 | 暖缓存已证明有效，但冷编和缓存分发仍是部署成本；graph partition 需先完成能力评估 | [Stage5 TODO-VA-2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>)、[Stage6 启动记录](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |

### 4.2 有收益但仍需保持回归的方向

| 类型 | 方向 | 历史结果和保留事项 | 来源（定位） |
|---|---|---|---|
| 性能/线性注意力 | Qwen3-Next、Qwen3.5 等混合线性注意力 | Stage 6 中 Qwen3-Next TP4 多负载提升约 11%～26%，Qwen3.5 部分负载提升 6%～14%；这是优势场景，应加入性能回归而非当作普遍倍率 | [Stage6 结论](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#六结论与开箱建议终版>) |
| 性能/缓存 | FxGraphCache 暖启动 | Stage 5/6 已证明跨重启缓存价值；需要缓存版本隔离、损坏缓存回退和分发测试，不能继续写“暖启动完全未验证” | [Stage6 启动记录](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |
| 性能/自定义算子 | 未设置时 `custom_ops=["all"]` | 8B/W8A8 历史上明显优于 none；显式用户配置仍应尊重，需要防止上游默认变化造成回退 | [platform.py:423](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L423>) |

## 5. 测试覆盖与工程缺口

### 5.1 模型和运行场景覆盖

| 编号 | 类型 | 覆盖缺口 | 当前状态和边界 | 来源（定位） |
|---|---|---|---|---|
| PI-VA-01 | 覆盖/P1 | 编译缓存跨启动加载的异常路径 | 普通 FxGraphCache 暖启动已验证；旧/半损坏缓存、AOT artifact 加载和硬 assert 回退仍缺完整覆盖 | [VA 章 PI-VA-01](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-03 | 覆盖/P1 | Multi-LoRA、热加载与 Inductor 组合 | FULL 挂死是前置阻塞；Meta 修复不代表组合覆盖完成 | [VA 章 PI-VA-03](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[bug O-01](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| PI-VA-04 | 覆盖/P1 | PD 分离及 KV connector 组合 | 只有冒烟和胶水探针，缺双向组合与异常路径；示例零实传问题见 bug O-13 | [VA 章 PI-VA-04](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-05 | 覆盖/P1 | EPLB × Inductor | 专家重排与编译缓存、ACL 图参数的组合未证 | [VA 章 PI-VA-05](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-12 | 覆盖/P2 | 多模态 encoder 编译 | 当前模型记录不能外推到 `compile_mm_encoder` | [VA 章 PI-VA-12](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-13 | 覆盖/P2 | 非默认 dynamic shapes | experimental 配置缺测试，动态 shape 粘滞标记问题见 bug T-04 | [VA 章 PI-VA-13](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| PI-VA-17 | 覆盖/P2 | 纯 FULL/FDO 大模型编译成本对照 | 单图可行性已证，但大模型编译时长和与保留切分的对照不足 | [VA 章 PI-VA-17](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| SPEC-REMAIN | 覆盖/P2 | suffix 和其他推测解码方法 | Eagle3/DFlash/ngram 已有代表验证；suffix 曾受 arctic-inference 依赖阻塞，其他无本地配方的方法不能计为通过 | [功能现状小结 V1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/00_功能设计现状小结.md>) |
| QUANT-REMAIN | 覆盖/P1 | FP8、MXFP8、W4A8、W4A4、compressed-tensors 全矩阵 | 部分受 eager 底层故障、权重缺失或显存限制；必须区分跳过、xfail、冒烟和数值通过 | [功能现状小结 V2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/00_功能设计现状小结.md>)、[量化 bug 汇总](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| A5-COVERAGE | 覆盖/P0 | A5 全模型、量化、多卡、精度和性能验证 | 当前主要是 Qwen3-0.6B 构建与冒烟，不能外推到完整支持 | [适配现状：A5](../vllm-ascend-Inductor适配现状.md)、[A5 部署记录](../a5-25-cann920-env-deploy.md) |

### 5.2 CI、版本和社区工程

| 编号 | 类型 | 工程缺口 | 当前状态和验收方向 | 来源（定位） |
|---|---|---|---|---|
| PI-TN-23 | 覆盖/P2 | AscendForge 黑名单回归轴 | 历史 75 条可解析中 14 passed/59 failed，另有 38 条语料漂移；需逐条归因并纳入例行门禁 | [torch 章 PI-TN-23](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| PI-3P-04 | 覆盖/性能工程 | mspti 预热、缓存分发、performance-mode 和 custom_ops 对照轴 | 性能测试基建尚未完全门槛化 | [第三方章 PI-3P-04](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>) |
| TODO-VA-7 | 覆盖/CI | eager、fusion_pass、inductor 三轨性能基线 | 有 benchmark 工具，但缺少稳定 CI 门槛；legacy 崩溃清理问题也需先处理 | [Stage5 TODO-VA-7](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| TEST-QUALITY | 覆盖/CI | 测试名称与实际路径不一致、缓存未隔离 | 当前源码仍可确认，先修测试再谈覆盖率；详细位置见 bug T-01～T-03 | [bug 测试问题](../bugfix/Inductor适配Bug定位与修复追踪.md#8-测试代码和测试环境自身的缺陷) |
| A5-CI | 覆盖/CI | A5 PR CI 未接入这些 Inductor 用例 | 需要明确模型集合、运行资源、缓存策略、精度和性能基线 | [适配现状：CI](../vllm-ascend-Inductor适配现状.md) |
| VERSION-MIGRATION | 工程/P0 | 0.27.1 到 0.30/后续 0.33 的移植 | 多个核心文件与上游冲突；先验证新 Triton/CANN，再分批合并和全量复验 | [升级评估](../04-升级评估.md)、[当前情况](../../手工文档/当前情况.md) |
| DEPENDENCY-MATRIX | 工程/P0 | 依赖声明与已验环境不一致 | 仓库/CI 仍有 triton-ascend 3.2.2，而 A5 记录使用 3.6.0.post2；需要明确支持组合和默认安装版本 | [pyproject.toml:21](<../../../inductor-meta-worktree/vllm-ascend/pyproject.toml#L21>)、[A5 部署记录](../a5-25-cann920-env-deploy.md) |
| PI-3P-03 | 工程/上游化 | 本地修复和 Pass 改动的社区 PR/MR | 需要逐项提交并回填链接，避免本地补丁随版本漂移失效 | [第三方章 PI-3P-03](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>) |
| PI-3P-05 | 工程/文档 | 用户可见的版本支持矩阵 | 记录 torch、torch_npu、CANN、Triton、vLLM、硬件和已验模型组合，不能只靠环境脚本反推 | [第三方章 PI-3P-05](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>) |

## 6. 源文档仍写 open、但后续已解决或状态已变化的事项

这些条目不能被重新当作当前 open 工作直接启动，应先更新源台账状态。

| 原编号 | 类型 | 原问题 | 后续事实和当前判断 | 来源（定位） |
|---|---|---|---|---|
| PI-VA-11 | 功能缺失/已解决 | `ir_enable_torch_wrap` 被强制 False | `be950e9cf` 已取消强钉，恢复上游 Inductor 派生；原 open 已过时 | [VA 章 PI-VA-11](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[platform.py:417](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L417>) |
| PI-VA-26 | 性能/已追平 | W8A8 decode 历史 0.75× | Stage 5 的 custom_ops 策略和栈演进后，Stage 6 同日对拍约 1.01×；不能继续沿用旧 0.75× 作为当前短板 | [VA 章 PI-VA-26](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[Stage6 头条表](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| PI-VA-27 | 性能/已落地 | FULL_AND_PIECEWISE 未成为 Inductor 默认 | `477377870` 已让未设置图模式跟随 -O 预设，O2/O3 默认 FULL_AND_PIECEWISE | [VA 章 PI-VA-27](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[graph_mode.md:47](<../../../inductor-meta-worktree/vllm-ascend/docs/source/user_guide/feature_guide/graph_mode.md#L47>) |
| PI-VA-28 | 性能/已执行 | DeepSeek 性能摸底空白 | Stage 6 已给出 V2-Lite BF16/W8A8 等数据；当前遗留转化为 MLA BF16 decode 优化，而不是“没有摸底” | [VA 章 PI-VA-28](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[Stage6 结果](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |
| PI-VA-01 | 覆盖/部分解决 | 跨启动缓存完全未验证 | 普通 FxGraphCache 暖启动已有证据；只保留 AOT、损坏缓存和异常回退子项 | [VA 章 PI-VA-01](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[Stage6 启动记录](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |
| PI-VA-02 | 可观测性/主体已解决 | Breakable CUDAGraph 静默使 Inductor 失效且缺引导 | `166027991` 和文档已增加 warning 与显式 `=0` 逃生门；剩余是用户体验评审，不应再写“完全无提示” | [VA 章 PI-VA-02](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)、[graph_mode.md:67](<../../../inductor-meta-worktree/vllm-ascend/docs/source/user_guide/feature_guide/graph_mode.md#L67>) |
| TODO-VA-6 | 功能/已关闭 | SwiGlu+Quant 的 aten pattern | F4 整体已由 `fe4fc22f0` 回退，本条随本体关闭；只有真实 all-linear-W8A8 目标链出现时才重开 | [Stage5 TODO-VA-6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| PI-TN-17 | 功能/已规避 | TE elide Pass 不执行 | VA wrapper 已恢复主轨优化，但 TE 根因未修；详细追踪移到 bug W-05，不应重复算一个未实现功能 | [torch 章 PI-TN-17](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)、[bug W-05](../bugfix/Inductor适配Bug定位与修复追踪.md) |

## 7. 建议执行顺序

1. **统一版本和依赖基线**：完成 VERSION-MIGRATION、DEPENDENCY-MATRIX，避免在旧栈上继续产生不可迁移结论。
2. **先修验证，再扩功能**：处理 TEST-QUALITY、A5-CI、A5-COVERAGE，并建立 eager 可通过的场景集合。
3. **关闭正确性前置阻塞**：LoRA、量化 eager 失败、缓存 hash 和数值归因按 bug 文档处理，再恢复相应 feature 覆盖。
4. **按实际模型需求选择底层能力**：AOT/cpp_wrapper、HOP、varlen、RNG、通信 lowering 不应无差别同时开发，应按目标模型触发排序。
5. **性能以同日同环境对拍为准**：优先 MLA BF16 decode、gather/GEMM kernel 和冷编，保留线性注意力及暖缓存优势。
6. **最后再评默认换轨**：只有默认配置正确性、覆盖和性能均满足要求后，才处理 TODO-VA-1。

## 8. 主要来源索引

- [功能遗留总表](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/04_总表.md>)
- [torch_npu / TE 功能与缺陷明细](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)
- [vllm-ascend 功能、性能与覆盖明细](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)
- [第三方与工程明细](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>)
- [Stage 5 TODO](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>)
- [Stage 6 性能结果与问题台账](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>)
- [0.30 升级评估](../04-升级评估.md)
- [当前版本与会议待办](../../手工文档/当前情况.md)
- [当前适配、测试和环境状态](../vllm-ascend-Inductor适配现状.md)
- [A5 部署记录](../a5-25-cann920-env-deploy.md)
- [Bug 定位与修复追踪](../bugfix/Inductor适配Bug定位与修复追踪.md)

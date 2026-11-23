# vLLM Ascend 与上游 Inductor 接入差距及 TorchNPU 新特性价值评估

## 结论

以官方 vLLM `v0.33.0`、当前 vLLM Ascend 工作树 `50153397a00a` 和 torch_npu `6f6a1bcc1217` 为准，主要差距不是基础的 `torch.compile` 调用，而是以下三类：

1. **启动与缓存能力**：standalone compile、AOT、Mega AOT 尚未形成完整可用的组合。
2. **图优化能力**：Inductor 图分区被禁用，ComboKernel 被禁用，上游多数融合 pass 没有 Ascend 对应实现。
3. **TorchNPU 新能力的配置传递**：TE 后端已有 C++ wrapper、VF Fusion、ComboKernel、kernel bundling 等能力，但 vLLM 侧缺少统一配置入口；其中不少仍属实验特性。

当前最值得优先做的是：

- **先补缓存身份和通用 TE 配置传递**，属于正确性与实验基础设施。
- **评估 standalone/AOT/Mega AOT 与 kernel bundling 的组合**，潜在价值主要是减少温启动时间，仍需实测。
- **性能工作优先定位实际慢 kernel 和有效融合覆盖**，不再预设新增调优或融合开关一定带来收益。
- **Inductor 图分区具有完整性和后续融合价值，但开发量和验证范围较大。**
- **TE ComboKernel 当前不支持 A5 和动态 shape，在近期 A5 工作中优先级不高。**

本文的代码差距分析未新增测试；性能价值同时引用本目录下已完成的真实模型实验。未实验项目只表示潜在价值，已实验项目按实际结果更新默认策略和优先级。

> 状态说明：本文区分当前工作树与 `Patch/` 下的阶段性实现。Coordinate Descent、MMRS 和 VF Fusion 的 patch 已完成真实模型实验，但这些修改不应被表述为已经进入当前 vLLM Ascend 工作树；是否合入及如何合入仍需单独决定。

## 一、相比官方 vLLM，还缺哪些接入

| 特性 | 当前状态与代码证据 | 价值判断 |
|---|---|---|
| **standalone compile / AOT / Mega AOT** | Ascend 对 standalone 显式报错，并默认关闭 AOT、Mega AOT：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:632`。官方在当前 PyTorch 版本下默认启用这些路径：`inductor-meta-worktree/.upstream-vllm-0.33.0/vllm/envs.py:379`、`:765` | **高，主要针对温启动**。可保存并直接恢复编译产物，减少 Dynamo tracing、FX 切图和重复编译。但当前 FX 缓存已经存在，所以不能理解成“完全没有编译缓存” |
| **Inductor 图分区** | Ascend 无条件重置 `use_inductor_graph_partition=False`：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:1402`。上游支持在 Inductor codegen 阶段分区：`inductor-meta-worktree/.upstream-vllm-0.33.0/vllm/config/compilation.py:672` | **中高，属于完整性和后续融合支撑**。让 pass 在完整图上运行，再切出 attention 等不安全区域，并支持 FULL 与 PIECEWISE 共用一次 Inductor 编译。它是 RoPE+KV cache、attention+quant 等融合的重要支撑路径之一，但不是绝对前置条件：上游也允许 `splitting_ops=[]` 配合 FULL graph。上游默认同样关闭，不属于默认行为回退 |
| **ComboKernel 水平融合** | 上游默认在非 CPU 平台开启：`inductor-meta-worktree/.upstream-vllm-0.33.0/vllm/config/compilation.py:986`；Ascend 强制关闭：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:564` | **目前中低**。可减少独立 pointwise kernel 的启动次数，但 TE 版本限制很多，见后文 |
| **单尺寸 `max_autotune`** | 上游对 `compile_sizes` 的单尺寸编译默认开启：`inductor-meta-worktree/.upstream-vllm-0.33.0/vllm/compilation/compiler_interface.py:769`。Ascend 默认关闭，但保留显式环境开关：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:652` | **待模型验证，不应预设为高收益**。只有存在有效候选、模板和可改进 shape 时才可能收益；NPU GEMM 大量走外部算子时，盲目开启可能主要增加编译成本 |
| **Coordinate Descent** | 当前工作树仍将 `coordinate_descent_tuning=True` 覆盖为 `False`：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:1691`。`Patch/CoordinateDescent/0001-allow-coordinate-descent-tuning-on-npu.patch` 已解除这一限制并保留显式环境开关 | **接入价值明确，默认性能价值低**。Qwen3-0.6B 和 Qwen3-32B 实验均无明确生成收益，小模型启动还增加 11.2%；保持默认关闭，详见 [专项实验](CoordinateDescent接入vLLM-Ascend.md) |
| **上游融合 pass** | Ascend 将上游融合标志全部置 False：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:114`；目前只重新加入 Ascend 的 norm+quant、QKNorm+RoPE、muls+add 和 TE 类型消除：`inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/ascend_post_grad_pass_manager.py:208` | **模型相关，中高**。缺口包括 activation+quant、attention+quant、RoPE+KV cache、MLA 相关融合、AllReduce+RMSNorm 等。不能直接打开 CUDA 实现，需要 NPU 算子和数值验证 |
| **Stock `torch.compile` / Dynamo trace-once 模式** | 除 `NONE` 和 `VLLM_COMPILE` 外都被降为 `NONE`：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:1859` | **低到中的完整性价值**。主要用于调试、问题隔离和备用路径，通常不是首选性能路径 |
| **TE 后端选择及配置传递** | `inductor_compile_config.npu_backend` 被接受但明确不生效，只能依靠进程级环境变量：`inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py:1776` | **中等完整性价值**。不妨碍单后端运行，但不利于按引擎/按编译选择后端，也使 TE 的实验配置缺少统一入口 |

### FixFunctionalizationPass 仍缺少 NPU 适配

Ascend 当前将上游 `FixFunctionalizationPass` 替换为 no-op，以规避 CUDA 专属算子引用：`inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/ascend_post_grad_pass_manager.py:162`。

上游该 pass 负责对 RoPE、RMSNorm、SiluMul 等自定义算子去函数化并消除冗余 copy：`inductor-meta-worktree/.upstream-vllm-0.33.0/vllm/compilation/passes/utility/fix_functionalization.py:19`。当前缺少的是 `_C_ascend` 对应算子的处理，主要影响 copy 数量和性能，不能直接归类为功能错误。后续应按真实图中的 auto-functionalized 节点逐项补充，而不是仅绕开 CUDA import。

### MMRS 已验证接入可行，但实测为负收益

`MatmulReduceScatterFusionPass` 已在旧的 Ascend graph 路径中按 SP 和硬件能力启用：`inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py:80`。当前工作树的 Inductor pass manager 尚未接入它。

`Patch/MMRS/` 下的阶段实现已经完成 Inductor 注入、Ascend SP 图形态、动态 token 维度和 padding 适配，并证明真实模型会调用融合算子。但是 Qwen3.8-27B TP2 实验中，生成中位数由 1.808825 s 增加到 2.431365 s，**延迟增加 34.42%，等效吞吐下降约 25.60%**；输出一致，开关两侧生成代码也确认了 MMRS 是否实际生效。

因此该工作的结论不是“优先推广融合”，而是：**保留默认关闭的实验能力，不以性能优化名义默认合入**。当前没有 profiler 证据，不进一步断定负收益来自融合算子内部、padding 还是图结构变化。完整过程和 patch 状态见 [MMRS 专项记录](MMRS接入vLLM-Ascend.md)。

## 二、TorchNPU 有、但 vLLM Ascend 尚未充分利用的新特性

### 1. Inductor/Triton kernel bundling 与 autotune 结果恢复

TorchNPU/TE 已支持：

- 将 TE 配置纳入原生 FX/AOT 缓存身份：`inductor-meta-worktree/pytorch/test/_inductor/triton_experimental/test_package.py:101`
- 通过上游 `TritonBundler` 注册并打包 NPU kernel 工件：`test_package.py:152`
- 跨进程恢复 autotune winner，避免重新跑完所有候选：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/npu_triton_heuristics.py:562`

这里复用的是 Inductor 原生 bundle/package 机制，不应理解为另有一个已经独立交付的“TorchNPU Dynamo package”接口。

**潜在价值较高，主要针对温启动和多实例部署，但组合收益尚未实测。** 当前 vLLM 的普通 FX 缓存已经可以自动利用其中一部分能力；真正缺失的是它与 vLLM standalone/AOT/Mega AOT 的完整组合。建议把二者作为一个项目，验证实际启动时间、跨进程恢复和缓存迁移，而不是单独增加另一套缓存。

### 2. TE C++ wrapper

TE 已注册自己的 C++ wrapper codegen：

- 注册入口：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/device.py:16`
- 测试覆盖普通编译和 mutation/downcast：`inductor-meta-worktree/pytorch/test/_inductor/triton_experimental/test_cpp_wrapper.py:297`

vLLM 的通用 `inductor_compile_config` 并没有明确禁止 `cpp_wrapper`，所以更准确的状态是：**底层能力存在，配置理论上可传入，但没有 vLLM+ACLGraph 的完整验证和默认策略。**

**价值：中等、实验性。** 主要减少 Python wrapper 和 kernel launch 的 CPU 开销，适合小模型、小 batch、CPU 受限场景；它不以降低编译时间为目标，还可能增加构建成本。

TE 的 C++ wrapper 要求同时启用 `triton.autotune_at_compile_time=True`：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/codegen/cpp_wrapper.py:292`。自动补该默认值只存在于 `torch.compile` 外层，而 vLLM 直接调用 `compile_fx`，不会执行这段逻辑：`inductor-meta-worktree/pytorch/torch_npu/utils/_dynamo.py:181`。因此目前不能只传 `cpp_wrapper=True`，还缺少 vLLM 侧默认推导和完整验证。

收益范围也需要限定：ACLGraph 重放不会再逐 kernel 经过 Python/C++ wrapper，所以主要候选场景是 eager、未捕获区域以及编译/捕获阶段，不能直接把它视为已捕获 decode 的稳态优化。

### 3. TE ComboKernel

TorchNPU 已不是“完全没有 ComboKernel 实现”：已有 `TEComboKernel` 和独立开关 `enable_te_combo_kernel`：

- 开关：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/config.py:452`
- 代码生成：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/codegen/triton.py:6409`

但准入条件非常窄：

- **A5 直接拒绝**：`codegen/triton.py:6337`
- 仅支持静态、同 shape、同 dtype、无依赖的 pointwise 成员。
- 动态 shape、reduction、C++ wrapper 等都会回退。

因此要使用它，vLLM 侧至少需要同时处理通用 `combo_kernels` 和 TE 自有开关。**对于当前 A5 目标，它暂时没有直接收益；对于非 A5 的静态尺寸编译才有实验价值。**

### 4. VF Fusion

TE 提供 `enable_vf_fusion`、`vf_fusion_mode`、`vf_merge_level` 等配置：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/config.py:432`。但当前编译器实际消费的是 `vf_fusion_mode`；旧 `enable_vf_fusion` 是兼容字段，会被忽略，不能用作生效证据。

`Patch/VFFusion/0004-expose-vf-fusion-mode-on-npu.patch` 已在 vLLM Ascend 增加 `ascend_compilation_config.vf_fusion_mode`，允许 `ub-aware-op` 和 `max-parallel`，并证明 mode 会进入真实 Triton 编译。该 patch 不属于当前工作树。

Qwen3.8-27B 单卡实验中，`max-parallel` 相对默认 `None` 的生成中位数增加 **0.11%**，属于未观察到可区分的变化；单次启动增加 **3.22%**。因此目前只有接入完整性价值，没有默认性能收益证据。默认继续使用 `None`，只有特定 kernel 或模型出现稳定收益时才按场景启用。该结论只覆盖已测 mode 和负载，不外推到所有 VF Fusion 配置。详见 [VF Fusion 专项记录](VFFusionMode接入vLLM-Ascend.md)。

### 5. Permute-gather / FlexAttention

- Permute-gather 针对特定非连续 reduction 访存模式，默认关闭：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/config.py:196`
- TE 也接入了 FlexAttention。

二者对一般 decoder-only LLM 的优先级较低：主流 vLLM attention 目前主要走自定义 attention 算子，并被切出 Inductor 编译区域。它们更适合特定 encoder、推荐模型或未来真正使用 FlexAttention 的模型，不建议因为“TorchNPU 有”就优先接入 vLLM。

### 6. DynamicNormQuant view 变体没有独立命中

`fuse_norm_quant_dynamic` 对应的 view/reshape 专用 pattern 已在 Inductor 路径存在，默认关闭。Qwen3.6-27B-W8A8 和 Qwen3-0.6B-W8A8 实验中，总命中只来自已有 direct pattern，view 专用 pattern 没有独立命中。

原因是上游 `NoOpEliminationPass` 先删除等形 view，后续 direct pattern 已能覆盖实际图形态。因此不能把开关两侧的耗时变化归因于该变体，也不能据此否定仍在真实生效的 direct 融合。当前应保持默认关闭；只有未来模型出现不能被提前删除的非等形 view 时，才需要重新评估。详见 [DynamicNormQuant 专项记录](DynamicNormQuant_view变体验证.md)。

### 7. 通用 NPU FX pass 尚未进入 vLLM pass 链

default NPU 后端会安装 `AscendCustomPrePass` / `AscendCustomPostPass`：`inductor-meta-worktree/pytorch/torch_npu/_inductor/__init__.py:323`。这些 pass 包含 cast/cat/view 折叠、batch embedding、grouped matmul、multi-slice-concat、fused matmul+ReLU 等模式，实现位于 `torch_npu/_inductor/fx_passes/ascend_custom_passes/`。

vLLM 会用自己的 pre/post pass manager 替换相应全局槽位；当前 Ascend 只补回 TE elide pass，没有链接这套通用 NPU pass。这是上游 vLLM 不会提供的独立增强机会，但不应整组默认打开：应先按目标模型检查图命中率、数值和性能，再选择有收益的 pass 接入，并保留其 UUID/缓存身份。

## 三、缓存正确性还有一个需要先补的问题

TorchNPU 原生 TE 路径安装了 `TeSourcePinPass`，将以下内容纳入缓存身份：

- TE Python 源码摘要；
- TorchNPU 版本；
- Triton Ascend 版本。

见 `inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/fx_passes.py:394` 和 `:472`。

但 vLLM 会用 `AscendPostGradPassManager` 覆盖全局 post-grad pass 槽；Ascend 目前只重新加入了 elide pass，没有加入 `TeSourcePinPass`。因此，**TE 后端源码变化导致的缓存失效保护，在 vLLM 路径上没有完整保留下来。**

这主要是正确性和开发效率问题，不直接提升推理性能，但应该是推广 AOT/package 缓存之前的前置工作。

## 四、不要重复算作“未接入”的能力

以下能力已经通过上游公共代码或 TorchNPU runtime 自动进入当前路径：

- 动态 shape、`compile_sizes`、compile ranges。
- 通用 `inductor_passes`。
- piecewise/full ACLGraph 包装。
- 本地 FX 编译缓存。
- TE 静态 launcher/vectorcall fast launcher 的运行时入口：`inductor-meta-worktree/pytorch/torch_npu/_inductor/triton_experimental/npu_triton_heuristics.py:672`。static launcher 仍需配置开启且 kernel 满足准入条件，fast launcher 在选定最终 launcher 后替换稳态调用入口；不能把“入口已存在”理解为所有 kernel 都必然走该路径
- TE 本地 autotune 缓存。
- 默认开启的 TE codegen、decomposition 和 reduction 优化。

这些可能仍缺模型覆盖或默认策略验证，但不是需要重新开发的接入点。

## 五、已有实验对性能优先级的修正

以下结果均来自本目录专项记录及其引用的 `server-results/inductor-feature/20261122_140509/` 原始数据。它们只约束已测模型、负载和配置，不外推为所有场景必然无收益。

| 特性 | 实测摘要 | 当前建议 |
|---|---|---|
| [Coordinate Descent](CoordinateDescent接入vLLM-Ascend.md) | Qwen3-0.6B 生成基本持平、启动增加 11.2%；Qwen3-32B 生成慢 0.21%，单次启动差异不能证明收益 | 功能 patch 已验证，可保留显式入口；默认关闭，不再作为近期性能重点 |
| [VF Fusion](VFFusionMode接入vLLM-Ascend.md) | Qwen3.8-27B 的 `max-parallel` 生成中位数慢 0.11%，启动增加 3.22% | 默认 `None`，仅作实验选项；其他 mode 仍需独立证据 |
| [MMRS](MMRS接入vLLM-Ascend.md) | Qwen3.8-27B TP2 生成延迟增加 34.42%，等效吞吐下降约 25.60% | 明确负收益；默认关闭，不以性能优化名义推广或默认合入 |
| [DynamicNormQuant view 变体](DynamicNormQuant_view变体验证.md) | 两个 W8A8 模型均无 view 专用 pattern 独立命中 | 无增量价值证据；保持默认关闭，保留已有 direct 融合 |

这些结果说明：**接入完整性与性能价值必须分开评价。** 功能真实生效不等于默认开启合理；没有独立命中时，开关两侧的耗时差异也不能作为特性收益或回退证据。

## 推荐顺序：按可量化问题推进

后续开发不应继续按“还有哪些开关没有接入”排列，而应围绕两个可量化目标：**缩短温启动时间**和**消除目标模型的实际性能回退**。接入完整性仍有价值，但不再自动等同于性能优先级。

### 1. 先完成缓存身份与正确性

把 TE 源码/版本指纹接入 Ascend pass UUID，并验证修改 TE 源码后 vLLM 缓存确实失效；同时检查新增编译配置是否进入 cache key。

这是小规模前置工作。否则后续 AOT、standalone 和跨进程缓存分发越多，陈旧缓存越难排查。

### 2. 主线做 standalone/AOT 最小可行实现，但分阶段决定投入

**第一步先验证普通 AOT 直接加载。** 当前 `VLLM_USE_AOT_COMPILE=1` 并未被禁止，只是跨启动加载尚未完整验证。先用 Qwen3-0.6B 检查：

- 首次启动能保存完整产物，新进程能直接加载；
- 输出一致，spawn worker 中可用；
- 与现有 FX Graph Cache 温启动相比是否有稳定、可重复的额外收益。

**第二步再适配 NPU standalone compile。** 只有普通 AOT 证明存在额外价值后，才继续处理 NPUBIN/Triton kernel 打包恢复、autotune winner、动态 shape、多个 piecewise 子图、TP2 和 ACLGraph。当前 Ascend 对 standalone 的直接拒绝，应在这些底层能力完成后移除，而不是先放开开关。

**第三步才评估 Mega AOT。** 只有 standalone 已稳定，并且 profiling 证明 Dynamo tracing、FX 拆图仍占明显启动时间，才继续投入 Mega AOT。

这一路线的验收标准不是“AOT 能运行”，而是**相对现有 FX 缓存温启动有明确的额外收益**。若普通 AOT 已没有额外收益，或 standalone 相对普通 AOT 没有进一步改善，应停止扩大这条路线。

### 3. 性能工作以 profiler 证据选择具体任务

选择 DeepSeek-V2-Lite BF16 decode 等已有回退记录的模型，或当前最重要的生产模型，固定相同负载采集 eager 与 Inductor 的 device timeline，定位新增或变慢的 kernel、额外 copy 和图间开销。

优先只处理排名前一两项，例如 gather/离散访存、GEMM/reduction tiling、多余 transpose/clone、内存分配或 wrapper 开销。每项改动都同时要求单 kernel 和端到端证据，避免用局部提升代替模型收益。

`FixFunctionalizationPass` 和通用 NPU FX pass 也按这一原则处理：只有 timeline 或 FX 图证明存在对应 copy/融合机会后才开发，不因为上游存在相同模块就直接补齐。

### 4. 条件成熟后再做通用基础设施和复杂图改造

- **通用 TE 配置传递**：等下一个确实需要暴露的后端配置出现时，再通过类型化 additional config 和逐次 `config.patch` 实现，同时处理 spawn、cache key 与多引擎隔离；它是实验基础设施，不预设性能收益。
- **Inductor 图分区**：当 standalone/AOT 可行性已经明确，或出现具体的跨 attention 融合、FULL 与 PIECEWISE 重复编译需求时，再单独立项验证 TE 支持度和编译成本。
- **C++ wrapper**：先补 `triton.autotune_at_compile_time` 的配置推导，再验证 eager/未捕获区域和 ACLGraph 组合；不预设它能改善已捕获 decode 的稳态重放。

### 5. 已验证无默认收益的特性不再占用近期性能开发名额

Coordinate Descent、VF Fusion、MMRS 和 DynamicNormQuant view 变体保持默认关闭。若未来重开，需要出现新的模型/shape 需求或 profiler 证据，并设定新的实验假设，而不是重复已有开关实验。

TE ComboKernel 建议等 A5 和动态 shape 支持后再提高优先级。

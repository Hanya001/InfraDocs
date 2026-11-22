# vllm-ascend Inductor 适配 Bug 定位与修复追踪

## 1. 文档目的与口径

本文汇总本次 vllm-ascend Inductor 适配过程中已经记录的缺陷，追踪其现象、定位证据、根因、修复或规避提交，以及当前状态。

代码核对基线：

- 仓库：`inductor-meta-worktree/vllm-ascend`。
- 分支：`hanya/feature/inductor_backend-1109`。
- HEAD：`81f53624fe81cf14c7a13ddace98e67c88033402`。
- 同时参考 `inductor-meta-worktree/docs` 中的设计文档、遗留总档、复现脚本和历史验证记录。

状态定义：

- **已修复**：已有明确代码修复和对应验证记录。
- **已修复，待闭环确认**：故障路径已消除并有验证，但总档仍要求进一步确认。
- **临时规避**：当前可避免触发，但底层根因未修，规避逻辑仍在代码中。
- **部分修复**：已修复一个已确认环节，但完整问题仍未闭环。
- **未解决**：没有最终修复，或历史修复已经回退。
- **待复测**：历史环境曾出现，后续环境有未复现或版本变化，不能直接认定为当前缺陷。
- **线索待确认**：只有会议或观察记录，尚缺独立复现和根因证据。

本文不将纯功能缺失、性能不足和测试覆盖不足一律记为 bug。量化 eager 基线失败、near-tie 数值差异等条目保留，是因为它们阻塞缺陷判定或可能涉及真实错误；在根因查明前，不预设其一定属于 Inductor。

**本文以源码、提交和历史证据的静态整理为基础，并持续补充后续 A5 实测。R-01 已新增 NPU 端到端验证；其他条目的“通过”“失败”“复现”仍指各自已有记录，不表示本文统一重新执行了全量测试。**

## 2. 状态总览

总览中的“类型”用于区分正确性、崩溃、兼容性、构建、稳定性、工程和测试问题；“来源”均为可点击链接，链接文字同时给出文档内编号或代码行号，便于跳转后定位。

| 编号 | 类型 | 缺陷 | 当前状态 | 修复/规避及后续提交 | 来源（定位） |
|---|---|---|---|---|---|
| F-01 | 正确性/编译 | Eagle3 下 RoPE fake 别名声明错误导致 `setStorage` 越界 | 已修复 | `7a75f23c7` | [Stage4 R16](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/CHECKPOINT.md>) |
| F-02 | 功能缺陷/编译 | LoRA shrink 算子缺 Meta 实现，FakeTensor/AOT 追踪失败 | 已修复 | `01b3c90e0` | [Meta 实现](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/meta_registration.py>) |
| F-03 | 正确性/融合 | 第二个同配置 engine 的融合 pattern 集为空 | 已修复 | `417fa2f7e` | [当前注册逻辑](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/passes/base_pattern.py#L53>) |
| F-04 | 配置/推测解码 | ngram_gpu × Inductor 因单例携带轨身份而选错 `pass_key` | 已修复，待闭环确认 | `4ac7b6fe0`、`51dfc786d` | [PI-VA-20](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| F-05 | 崩溃/推理张量 | `_memoized_downcast` 读取推理态 `_version` 导致 EngineCore 初始化失败 | 历史分支已修复；当前工作区已复验，官方 master 未包含 | 历史 torch_npu `2cb1a3497f`；本次以 `is_inference()` 绕过缓存，0.8B/vLLM 0.33 已验证，尚未提交 | [Bug2 台账](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/torch-npu bug/bug2_memoized_downcast_推理态.md>) |
| F-06 | 兼容性/依赖 | 兼容 stub 遮蔽 triton-ascend 3.6 的真实 experimental 包 | 已修复 | `d1bc1d256` | [当前包检测](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/__init__.py#L22>) |
| F-07 | 构建/A5 | MLA Prolog v3 日志宏损坏导致 Ascend950 构建失败 | 已修复 | `81f53624f` | [A5 部署问题 6/7](../a5-25-cann920-env-deploy.md) |
| F-08 | 构建/版本 | PyTorch 2.13/CANN 头文件和 C++ 标准导致构建失败 | 已修复 | `9e3caae09`、`6b730dc50`、`9a76a2372` | [升级评估](../04-升级评估.md) |
| F-09 | 兼容性/融合 | graphex QKNorm 历史 `ERR03001` | 新栈已自愈，待持续回归 | torch_npu 升级后复验通过，无单独修复提交 | [功能现状小结](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/00_功能设计现状小结.md>) |
| F-10 | 编译/初始化 | TE 懒激活晚于 AOT decomposition，触发 `_to_copy` fallback/decomp 冲突 | 已规避 | worker 编译前导入 `torch_npu._inductor` | [断点地图 6.1](<../../../inductor-meta-worktree/vllm-ascend/docs/source/developer_guide/performance_and_debug/inductor_track_breakpoint_map.md#6-known-breakpoints-and-traps-living-list>)、[worker.py:434](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/worker/worker.py#L434>) |
| F-11 | 配置/编译 | 非法 `graph_fusion_manager` 键进入 Inductor `config.patch` | 已修复 | Inductor 轨使用 `post_grad_custom_post_pass` | [断点地图 6.3](<../../../inductor-meta-worktree/vllm-ascend/docs/source/developer_guide/performance_and_debug/inductor_track_breakpoint_map.md#6-known-breakpoints-and-traps-living-list>)、[platform.py:130](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/platform.py#L130>) |
| F-12 | 工具/基准 | 随机 prompt 超出 DeepSeek 词表 | 已修复 | 按模型 `vocab_size` 计算 `model_vocab_safe` | [Stage6 #3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>)、[脚本:100](<../../../inductor-meta-worktree/docs/vllm/env/scripts/bench_inductor_perf.py#L100>) |
| F-13 | 环境/依赖 | 缺少 `model_hosting_container_standards` 导致 CLI 导入失败 | 已解决 | 安装 0.1.16 | [Stage6 #1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| W-01 | 数值/量化 | 动态 Norm+Quant 融合与 eager 非逐 token 一致 | 临时规避 | `b836b75d1` 默认关闭新动态融合 | [PI-VA-18](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| W-02 | 数值/底层算子 | GDN 融合算子正常输入下数值损坏 | 临时规避，底层未修 | `44c58bc72` 加强探针并回退 FLA | [Bug1 台账](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/torch-npu bug/bug1_npu_chunk_gated_delta_rule_数值破坏.md>) |
| W-03 | 编译/Pass | `FixFunctionalizationPass` 在 NPU 上崩溃 | 临时规避，根因未闭环 | `AscendPostGradPassManager` 将其替换为 no-op | [PI-VA-08](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| W-04 | 优化/融合 | QKNorm+RoPE 在 Inductor 真图上 0 命中 | 部分修复，不影响正确性 | `44c58bc72` 修注册时序；真图形状差集仍在 | [PI-VA-19](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| W-05 | 编译/Pass 注入 | TE elide Pass 被 vLLM `config.patch` 替换语义遮蔽 | 临时规避 | `417fa2f7e` 增加 VA wrapper，TE 注入机制未改 | [PI-TN-17](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)、[PassManager:59](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/ascend_post_grad_pass_manager.py#L59>) |
| O-01 | 稳定性/LoRA | LoRA 混合请求在 FULL 图回放前挂死 | 未解决 | `num_active_loras` 修复试验无效且未合入 | [PI-VA-03](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| O-02 | 数值/对拍 | MoE/W8A8/PCP 等 near-tie 数值分叉 | 未解决，待归因 | 多个对拍用例仍为 strict xfail | [PI-VA-25](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| O-03 | 崩溃/量化 | FP8 block 原生权重 eager 加载报 561103 | 未解决 | 待区分 VA scale 处理与 CANN 拷贝核问题 | [PI-VA-14](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| O-04 | 崩溃/量化 | MXFP8 eager `profile_run` 报 161001 | 未解决，外部阻塞 | 指向 `aclnnDynamicMxQuantV2` | [PI-TN-15](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| O-05 | 稳定性/量化 | DSV4 W4A8 冒烟收尾段 AICore/HCCL 超时 | 未解决，待定界 | 生成判据曾通过，尾错尚需单独定位 | [PI-TN-16](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| O-06 | 稳定性/缓存 | MoE TP4 冷编译偶发 `.ttir` 文件消失 | 未解决，低频 flake | 怀疑并行预编译与缓存清理竞态 | [PI-VA-23](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| O-07 | 崩溃/legacy | legacy `fusion_pass` 的 AOTAutogradCache 断言 | 修复后回退，当前仍有历史风险 | `417fa2f7e` 修复，`2f426bb16` 回退 | [Stage6 #4](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| O-08 | 配置/legacy | inproc legacy 轨 AscendConfig 双来源导致 #65 | 修复后回退，当前仍有历史风险 | `146f0007c` 修复，`ce8e97afd` 回退 | [Stage4 TODO](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/TODO.md>) |
| O-09 | 正确性/算子契约 | max-pool indices 返回 int8 bitmask，违反 aten int64 契约 | 未解决，op-plugin 侧 | torch_npu 只能去分解，影响 14 条黑名单用例 | [PI-TN-07](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| O-10 | 崩溃/底层算子 | `aclnnSwiGluQuantV2` 动态模式 AIV/MTE 越界 | 未解决，独立上游候选 | F4 已由 `fe4fc22f0` 回退，不解决底层算子问题 | [Stage5 TODO-TN-5](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| O-11 | 兼容性/调试 | depyf 0.20.0 与 torch 2.13 签名不兼容 | 未解决，外部阻塞 | 禁用 `VLLM_DEBUG_DUMP_PATH`，走 `TORCH_COMPILE_DEBUG` | [PI-3P-01](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>) |
| O-12 | 稳定性/第三方 | Mooncake NPU 扩展退出时 glibc abort | 未解决，外部阻塞 | 测试暂按进程内结果验收 | [PI-3P-02](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>) |
| O-13 | 正确性/PD 示例 | PD 示例没有真正发生 KV 传输 | 未解决，示例缺陷 | 缺 router 侧 `kv_transfer_params` 注入 | [PI-VA-24](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>) |
| O-14 | 编译/legacy | `compile_fx` 接收但未转发 `decompositions` | 当前代码仍存在 | 断点地图写了修复方向，但当前 `aot_autograd` 调用未落实 | [断点地图 6.2](<../../../inductor-meta-worktree/vllm-ascend/docs/source/developer_guide/performance_and_debug/inductor_track_breakpoint_map.md#6-known-breakpoints-and-traps-living-list>)、[compiler_interface.py:39](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/compilation/compiler_interface.py#L39>) |
| O-15 | 稳定性/多卡 | TP2/TP4 EngineCore 出错后父进程等待 RPC 挂死 | 未解决 | 关闭 autograd cache 只避开触发源，清理路径未修 | [Stage6 #7/#9](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| O-16 | 工具链/spawn | `vllm bench startup × fusion_pass × spawn` 报 `NoneType.record_statistics_debug_info` | 未解决 | 以自有 init 时间口径替代，未修工具链 | [Stage6 #6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| O-17 | 构建/AscendNPU-IR | `hivmc-a5` 链接缺少 `isManualVFScope` 等符号 | 未解决，外部阻塞 | 不影响当前 `bishengir-compile` 主链，旁支工具未修 | [A5 部署遗留 2](../a5-25-cann920-env-deploy.md) |
| O-18 | 数值/RMSNorm | A5 Qwen3-0.6B 原生与 Inductor 分解 RMSNorm 的数值差在 near-tie 处翻转 argmax | 未解决，根因已定位 | 实验确认保留原生 RMSNorm 可恢复与默认/eager 一致；生产修复待定 | [专项定位](20261112_Qwen3-0.6B-RMSNorm数值差异定位.md) |
| O-19 | 崩溃/量化 | A5 Qwen3.5-35B-A3B-W8A8-MTP 的 INT8 MoE 融合算子收到单个 `Parameter`，而接口要求 `Tensor[]` | eager跑不通 暂不解决 | 默认、Inductor 和 eager 均在 profiling 中复现；另有 `swiglu_limit`、`use_mxfp_quant` 无效关键字，当前未修改源码 | [专项定位](20261117_Qwen3.5-35B-W8A8-MTP融合量化算子参数错误.md) |
| O-20 | 兼容性/底层算子 | A5 DeepSeek-V4-Flash-FP8 在 TP4 下将 64 个 Q head 切为每 rank 16 个，但 FP8 KV sparse-attention 仅支持 64/128 | 暂不解决 | 默认、Inductor 和 eager 均有 `got 16` 直接证据；不通过调整 TP 或删除校验规避，BF16 KV 替代路径尚未实测 | [专项定位](20261117_DeepSeek-V4-FP8-TP4稀疏注意力Head数限制.md)、[社区 #21395](https://github.com/vllm-project/vllm-ascend/issues/21395) |
| O-21 | 数值/一致性 | A5 Qwen3.8-27B-W8A8 的 eager、default 和 Inductor 固定输入及 logprob 均存在路径差异 | 未解决，定位完成，无保留修复 | default/eager 本身也不一致；sigmoid 与 RMSNorm 对照可消除 default/Inductor 差异，但全局 sigmoid 原型已撤回；CANN NormQuant 融合的 eager 差异仍需单独处理 | [专项定位](20261122_Qwen3.8-27B-W8A8精度定位.md) |
| O-22 | 启动/量化 | A5 Qwen3.6-35B-A3B-W8A8 的旧 NZ MoE 融合算子缺少动态 shape 配置，eager/default/Inductor 均在 profiling 中失败 | 当前环境已修复 | vLLM-Ascend `7db008b78` 切换到 `npu_grouped_matmul_swiglu_quant_v2`，保留融合；`87fe251e7` 的小尺寸真实 NPU MoE 测试通过 | [专项定位](20261123_Qwen3.6-35B-W8A8-MoE融合算子V2启动修复.md) |
| R-01 | 正确性/编译 | Qwen3.5 的 GDN 后置 LayerNorm 被 TE fallback 吞掉，随后 `eps` 又被推导为 FP64 | 本地验证通过，尚未合入分支 | 保留 `triton_kernel_wrapper_mutation` 专用 lowering，并在 TE wrapper 中让用户 kernel 的 Python float 使用 FP32；0.8B/vLLM 0.33 冷编译和缓存复跑通过，未再出现连续 `!` | [专项定位](20261121_Qwen3.5-GDN-LayerNorm编译问题定位.md)、[PI-TN-18](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>) |
| R-02 | 兼容性/监控 | 旧 triton-ascend 的 `triton.knobs` 导致 JIT monitor 失败 | 旧栈临时规避，新栈待复测 | `f4bce8cd1` 容错降级 | [Stage5 TODO-3P-1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |
| R-03 | 性能/launcher | A5 Qwen3.8-27B-W8A8 的 RMSNorm 特化参数仍被 C launcher 打包，设备参数缓冲区与 TTIR 签名不一致 | 本地验证通过，尚未合入分支；上游专门 PR 未合入 | Triton-Ascend `make_launcher` 在设备打包时将 `constants` 参数视为 `constexpr` 跳过，Python 侧仍保留完整参数检查；常量特化和原融合/grid 全部保留，算子恢复至约 0.0895 ms | [专项定位](20261119_Qwen3.8-27B-W8A8-RMSNorm常量特化性能回退.md)、[性能复测](../tests/20261119_Qwen3.8-27B-w8a8_32768_1024_profiling.md)、[上游 Issue #2407](https://github.com/triton-lang/triton-ascend/issues/2407)、[上游 PR #2387](https://github.com/triton-lang/triton-ascend/pull/2387) |
| R-04 | 正确性/编译 | TE linearize 将 `npu_triton_helpers.lt(tmp0, 0)` 的常量误改为 reduction 维度编号，导致 Qwen3.5 index kernel 错误 | 本地验证通过，尚未合入分支；官方 master 未包含 | `codegen/triton.py` 的 reduction 维度替换排除 `npu_triton_helpers.*`；与 F-05、R-01 联合完成 0.8B 冷编译和缓存复跑 | [专项定位](20261121_20261121_Qwen3.5-0.8B-TE-Inductor问题定位与修复.md) |
| U-01 | 缓存/待确认 | 不同后端可能复用编译缓存 hash | 线索待确认 | 目前只有会议记录，缺独立复现和完整归因 | [1105 会议](../../手工文档/1105会议.md) |
| E-01 | 权重/显存 | DSV4 名称是 W4A8、实体为 W8A8_DYNAMIC，TP4 加载 OOM | 未解决，受权重和硬件约束 | 与 O-05 收尾尾错分开；需正确权重或更多卡 | [Stage6 #8](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| E-02 | 交付物/环境 | A5 环境缺 cp312 vllm_ascend 预编译 wheel | 当时已规避 | editable 现场编译，正式 cp312 交付物仍需确认 | [A5 部署问题 2](../a5-25-cann920-env-deploy.md) |
| T-01 | 测试/配置 | 8B W8A8 默认图模式测试没有启用 Inductor | 当前源码仍存在 | 补 backend 并隔离缓存后重跑 | [测试:231](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_w8a8.py#L231>) |
| T-02 | 测试/配置 | MoE W8A8 TP4 的 `weight_nz_mode` 放置错误 | 当前源码仍存在 | 移到外层 AscendConfig 后重跑 | [测试:78](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/four_card/inductor/test_inductor_track_moe_w8a8_tp4.py#L78>) |
| T-03 | 测试/缓存 | 部分“冷编译”测试不清空固定缓存目录 | 当前源码仍存在 | 使用独立空目录并断言新增产物 | [长序列测试:81](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_long16k.py#L81>) |
| T-04 | 测试/动态 shape | `mark_dynamic` 标记随复用 tensor 污染后续 case | 测试陷阱，需规范用法 | 每个 case 使用新 tensor | [断点地图 6.7](<../../../inductor-meta-worktree/vllm-ascend/docs/source/developer_guide/performance_and_debug/inductor_track_breakpoint_map.md#6-known-breakpoints-and-traps-living-list>) |
| T-05 | 测试/显存 | Qwen3-Next 探针未固定 `gpu_memory_utilization` 导致 OOM | 已处置 | 探针显式设为不超过 0.80 | [Stage6 #9](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#五问题与初步分析台账易修则修不易修则记录>) |
| T-06 | 文档/调试 | 缓存检查命令缺少 `rank_0_0` 层级 | 文档问题 | 使用包含 rank 目录的实际缓存路径 | [A5 部署问题 8](../a5-25-cann920-env-deploy.md) |
| T-07 | 测试环境/依赖 | 缺 `pytest_mock` 导致 `test_comm_utils` 收集错误 | 环境问题，待规范依赖 | 安装 `pytest-mock`，不能只靠 ignore 规避 | [Stage5 TODO-3P-2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>) |

## 3. 已修复、已闭环及本地验证的问题

### F-01：RoPE fake 输出别名声明错误，Eagle3 触发 `setStorage` 越界

- **现象**：Eagle3 × Inductor 冷编译后，draft head 进入 attention 的 `reshape_and_cache` 时发生 storage 越界。
- **定位**：真实 RoPE 算子返回新分配的连续 tensor，但 `_rope_forward_oot_impl_fake` 直接返回 `(query, key)`，使 Inductor functionalization 误以为输出别名输入。生成 wrapper 因而用原 qkv view 的 stride  reinterpret 新输出，在较小 storage 上触发越界。
- **证据**：engine 复现、默认轨与 Inductor 轨 `output_code` 对比、最小 twin-op 复现三层证据一致；将 fake 改为 `empty_like` 后最小复现转绿。
- **修复**：`7a75f23c7`，fake 返回 `torch.empty_like`，与真实算子语义对齐。
- **验证记录**：Eagle3 冷编译通过，acceptance 在社区基线范围内；默认轨 acceptance 保持逐位一致。
- **当前状态**：已修复。

### F-02：LoRA `bgmv_shrink/sgmv_shrink` 缺 Meta 实现

- **现象**：编译区域包含 LoRA 时，FakeTensor 传播和 `aot_eager` 编译失败。
- **定位**：所有 LoRA forward 都经过 shrink 算子，但此前只有 expand 注册了 Meta；去掉或补上 Meta 的对照实验直接证明缺口。
- **修复**：`01b3c90e0`，在既有 capability gate 下注册 Python 侧 no-op Meta kernel。
- **验证记录**：补 Meta 后完整 LoRA 链可正常 trace，68 个 LoRA UT 通过。
- **当前状态**：已修复。该修复只解决编译追踪，不解决 O-01 的 FULL 图挂死。

### F-03：第二个同配置 engine 丢失全部融合 pattern

- **现象**：第一个 engine 可以融合，第二个同配置 engine 的 `PatternMatcherPass` 没有 pattern。
- **定位**：torch 的 `seen_patterns` 去重是 per-pass-instance，而旧代码用进程级集合跳过重复注册，导致新 pass 实例为空。
- **修复**：`417fa2f7e` 修改 `base_pattern.py`，每个 `PatternMatcherPass` 实例都执行 torch 注册；仅对进程全局的 npugraph_ex 注册表去重。
- **当前状态**：已修复。当前实现见 `vllm_ascend/compilation/passes/base_pattern.py`。

### F-04：ngram_gpu × Inductor 选错 `pass_key`

- **现象**：模型构造期回退到 legacy `graph_fusion_manager`，随后 `compile_fx config.patch` 因非法 Inductor 配置键报错。
- **历史定位**：AscendConfig 单例被另一个较早创建的 auto 对象换绑；`146f0007c` 的 drafter stub 只绕过部分路径，没有解决 Inductor 腿的换绑问题。
- **最终处理**：
  - `4ac7b6fe0`：`_inductor_track_active()` 改读上游 per-engine `get_current_vllm_config_or_none()`，轨身份不再由 AscendConfig 单例承载。
  - `51dfc786d`：恢复 ngram drafter 的上游构造器，移除临时 stub。
- **验证记录**：独立冷缓存 ngram_gpu × Inductor 探针通过，legacy 对照也通过；总档标为“已解决、待闭环确认”。
- **当前状态**：故障机制已绕开且有验证。原“单例被谁换绑”的机理不再影响该路径，不需要为使用 Inductor 而继续追查。

### F-05：`_memoized_downcast` 读取推理态 `_version`

- **现象**：vLLM serve 在 EngineCore `profile_run` 阶段抛出 `RuntimeError: Inference tensors do not track version counter`；进程内 `LLM()` 不一定触发。
- **定位**：`npu_triton_heuristics.py` 原代码只捕获 `AttributeError`，而推理态 tensor 读取 `_version` 会抛 `RuntimeError`。
- **修复**：torch_npu `inductor-for-vllm@2cb1a3497f`，异常捕获扩为 `(AttributeError, RuntimeError)`，推理态退回无缓存普通 cast。
- **验证记录**：修复前 serve 启动必崩，修复后历史记录为 550 秒就绪并可生成。
- **当前状态**：历史开发分支已有修复，但官方最新 `master@5eb53d43e5` 仍只捕获 `AttributeError`，不能记为官方主干已修复。
- **2026-11-21 最新进展**：Qwen3.5-0.8B / vLLM 0.33 的进程内 `LLM()` 路径也复现该问题。`a94d4eb291` 将第 0 维单张量 index 改走 TE kernel，int64 索引因此进入 downcast 缓存；三种 tiling 的 PreRun 均报同一 `_version` 异常。当前工作区改用 `src.is_inference()` 直接绕过缓存，最小复现及完整模型冷编译、缓存复跑均通过，尚未提交或推送。

### F-06：兼容 stub 遮蔽真实 `triton.experimental`

- **现象**：triton-ascend 3.6 已提供真实 `triton.experimental/gluon`，旧兼容 stub 却抢先注册同名模块，导致 Triton kernel 编译失败。
- **定位**：stub 原为 3.2.x 无 Gluon 的环境设计，未先检查真实包是否存在。
- **修复**：`d1bc1d256`，仅在真实 `triton.experimental` 不存在时创建 experimental stub，并单独判断 Gluon。
- **验证记录**：A5、CANN 9.2.0、torch_npu 2.13、triton-ascend 3.6.0.post2 下，冷缓存编译结果与 eager 一致，Qwen3-0.6B 冒烟生成成功。
- **当前状态**：已修复。

### F-07：Ascend950 构建时 MLA Prolog v3 日志宏损坏

- **现象**：Ascend950/A5 构建失败；910B 构建不会编译这些文件，因此此前未发现。
- **定位**：`6b730dc50` 为 CANN `const char*` 参数增加 `.c_str()` 时，部分宏调用出现括号不平衡、字符串内误放 `.c_str()`、return 被吞入宏参数等问题。
- **修复**：`81f53624f` 修正 14 个调用点，保留原 `const char*` 适配。
- **验证记录**：Ascend950DT、CANN 9.2.0 下 custom aclnn 和 `vllm_ascend_C` 构建通过，Qwen3-0.6B eager/Inductor 冒烟通过。
- **当前状态**：已修复。

### F-08：PyTorch 2.13/CANN 构建兼容问题

- **现象**：旧 CMake 强制 C++17 和 torch 版本门禁无法构建 torch 2.13；CANN/torch_npu 头文件及日志宏参数类型也不匹配。
- **修复**：
  - `9e3caae09`：C++17 升为 C++20，调整 torch 版本门禁。
  - `6b730dc50`：修复 CANN 日志宏参数和 torch_npu 2.13 ACL 类型声明兼容。
  - `9a76a2372`：声明 torch/torch_npu 2.13 及配套依赖版本。
- **验证记录**：910B2 上扩展构建、算子注册和 Qwen3-0.6B 离线生成通过。
- **当前状态**：已修复。F-07 是这一适配过程中后续暴露的 A5 专用回归，不能因 F-08 通过而省略 A5 构建验证。

### F-09：graphex QKNorm `ERR03001` 历史失败

- **现象**：旧栈 `graphex_qknorm` 用例曾报 `vllm.qkv_rmsnorm_rope ERR03001`。
- **定位**：stash 基线对照表明是存量问题，不是 Inductor 新改动引入。
- **修复/变化**：没有独立修复提交；torch_npu 升级后的复验中，两个相关用例均通过，记录为“新栈自愈”。
- **当前状态**：历史问题在新栈未复现，需要保留回归，而不能声称已定位并修复具体根因。

### F-10：TE 懒激活晚于 AOT decomposition

- **现象**：自定义入口在第一次 Inductor 编译前进入 AOT，出现 `both a fallback and a decomp for same op: aten._to_copy.default`。
- **定位**：环境变量选择的 TE 后端原本在第一次编译时才激活，晚于 AOT decomposition 和 fallback 检查。
- **规避**：worker 初始化时在 `torch.npu.set_device` 后主动 `import torch_npu._inductor`，提前完成后端加载；当前代码见 `vllm_ascend/worker/worker.py`。
- **当前状态**：主路径已规避。该初始化顺序需要保留回归，不能只检查环境变量是否设置。

### F-11：非法 PassManager 配置键进入 Inductor

- **现象**：`ConfigModule.patch` 报 `torch._inductor.config.<key> does not exist`。
- **定位**：legacy `graph_fusion_manager` 不是合法 torch Inductor 配置键，却被送入 `config_patches/inductor_compile_config`。
- **修复**：Inductor 轨的 `pass_key` 改为真实存在的 `post_grad_custom_post_pass`；legacy 轨仍使用原键。
- **当前状态**：已修复。F-04 是该通用问题在 ngram_gpu 单例换绑场景中的特殊触发路径。

### F-12：benchmark 随机 prompt 超出模型词表

- **现象**：DeepSeek-V2-Lite 报 `Token id 151313 out of vocabulary`，其词表上限小于 Qwen 使用的固定 `VOCAB_SAFE`。
- **修复**：benchmark 脚本根据模型 `config.json:vocab_size` 计算 `model_vocab_safe`，并保留缓存和确定性校验。
- **当前状态**：工具 bug 已修复，Stage 6 受影响负载已重发；不影响 vLLM 产品运行。

### F-13：vLLM CLI 缺少 `model_hosting_container_standards`

- **现象**：editable 重装后，vLLM CLI 的 api_server 导入链缺包。
- **处理**：安装 `model-hosting-container-standards==0.1.16` 后恢复。
- **当前状态**：环境问题已解决；部署文档和依赖检查应避免重复缺失。

### R-01：Qwen3.5 的 GDN 后置 LayerNorm 被 TE fallback 吞掉及 `eps` 类型错误

- **历史现象**：GDN 融合已禁用且 FLA marker 确认生效后，Inductor 轨仍输出重复异常字符，eager 打相同补丁后正常。早期候选为 `fused_gdn_gating_kernel` 或 FLA-under-track，Stage 6 另有未复现记录，因此当时只能标为待复测。
- **本次复现**：20261112 A5 模测中，Qwen3.5-2B 的 12 条提示词跨路径全部从第 0 个生成 token 分歧。Inductor 的 11 条输出为连续 128 个 `!`，另一条随后也退化为 `!`；默认路径文本连贯。Qwen3.5-35B-A3B 出现同型损坏。两条路径均使用 FLA，且 gating/FLA 的共有 `.npubin` 逐字节相同，排除了旧候选是本次主因。
- **根因一：LayerNorm 节点丢失**。Dynamo/AOT 将 GDN 后置的 `layer_norm_gated` 归约为 `triton_kernel_wrapper_mutation`。TE 的批量 fallback 注册覆盖了上游专用 lowering，通用 `FallbackKernel` 又假设 HOP 都是 functional，未登记 `Y/Rstd` 的 mutation。该 HOP 返回 `None`，于是被当成无输出、无副作用节点清理。最终 `output_code.py` 只分配未初始化 `buf0` 并直接送入后续 `unquantized_gemm`。默认路径则保留并执行同一 kernel。
- **根因二：`eps` 类型不匹配**。恢复 HOP 后，Inductor 的 `signature_of` 将 Python float 推导为 `fp64`，而 eager Triton-JIT 的 `mangle_type` 对同一参数推导为 `fp32`。Triton-Ascend 的 `op_dtype_config.py` 明确将 A2/A5 的 FP64 `sqrt` 列入 `unsupported`，所以出现编译失败；若强行在设备端转成 FP32，又会在链接阶段遇到 `undefined symbol: __truncdfsf2`。这不是 A5 所有 FP64 运算都不支持，而是当前 Triton-Ascend 路径明确屏蔽 FP64 `sqrt`。
- **最终修改**：在 `torch_npu/_inductor/triton_experimental/lowering_override_list.py` 中仅将 `_tkw.triton_kernel_wrapper_mutation` 加入 `KEEP_UPSTREAM_LOWERING`，恢复上游 `UserDefinedTritonKernel` 路线；在 `vllm_ascend/ops/triton/layernorm_gated.py` 中从 `do_not_specialize` 移除 `eps`，并改为 `eps: tl.constexpr`。官方 PyTorch 的 `wrapper.py`、`triton_utils.py` 均未修改，未使用 custom-op，也未保留 #190340 的产品补丁或 TE 注解读取扩展。
- **为什么可以用 constexpr**：当前 vllm-ascend 和上游 vLLM 的 `RMSNormGated` 使用点中，`eps` 都来自模型配置或模块初始化，未发现推理过程中动态修改。不同配置分别产生专用 kernel 和缓存项即可，不会按请求重复编译。若未来需要运行时动态 `eps`，应改用与 eager 一致的 FP32 签名，而不是简单恢复本特化。
- **最小 LayerNorm 对拍**：`eps is constexpr=True`，编译与 eager 的 `out` 和 `rstd` 最大绝对差均为 0。
- **端到端验证**：最终结果目录为 `/home/z00896543/scripts/bench213/a5/20261116_qwen35_2b_hop_constexpr_final/Qwen3.5-2B`。默认和 Inductor 退出码均为 0，12 条输出中 9 条逐 token 一致，Inductor 自身两次生成稳定，默认路径自身存在续写不稳定，未出现全零 token 或连续 `!`，三轮性能测试均完成。
- **生成代码证据**：最终签名为 `eps: constexpr` 且常量值为 `1e-06`；`_layer_norm_fwd_1pass_kernel_npu_0.run(...)` 位于后续 GEMM 之前，并生成对应 `.npubin`，不再存在空 buffer 直接进 GEMM 的情况。
- **当前状态**：本地临时方案已通过 A5 实测，原始灾难性生成损坏在当前修改下消失，但尚未合入任何分支，不能标记为已修复。跨路径只有 9/12 逐 token 一致，不能据此宣称两条路径达到完整数值验收，也不能把该结果外推为所有 Qwen3.5 或量化模型均已验证。W-02 的 GDN 融合算子数值缺陷是独立问题，不在本条修复范围内。
- **2026-11-21 最新进展**：`3c5c3eb375` 的 HOP 保护和 TE wrapper FP32 签名策略已适配当前 torch 2.13 / vLLM 0.33 工作区。只修 index 和 inference downcast 时，Qwen3.5-0.8B 仍连续输出 `!`；加入本项修复并完成全部相关修复后，冷编译、无插桩缓存复跑和最终复验均通过，输出正常。当前修改尚未提交或推送。
- **追踪提交**：#190340 `644154dd707f81ae7f07788dcfeb9e9f874e345e` 提供上游同类签名修复参考，已在 PyTorch 2.15 release 中包含；本次最终运行环境没有应用该提交，仅保留其 patch 和定位记录用于追踪。

### R-03：RMSNorm 特化参数的 C launcher 打包错误

- **现象**：Qwen3.8-27B-W8A8 在 A5、32768/1024 下，Inductor 输出吞吐为 131.25 tokens/s，default 为 162.76 tokens/s；平均 TTFT 和端到端耗时明显增加。
- **定位**：prefill 中 8192×5120 的 Inductor RMSNorm 融合内核平均 374222.77 μs，而 default 原生 RmsNorm 为 48.03 μs。慢快版本规范化 TTIR 的计算操作完全一致，TTAdapter/BCMLIR 也只有特化参数及 program-id 编号差异。构造相同 BCMLIR、相同 `.npubin` 哈希的两种签名后，仍分别出现 686.37 ms 和 0.0897 ms，证明设备指令不是根因。
- **根因**：Triton-Ascend `make_launcher(constants, signature, metadata)` 接收但未使用 `constants`。已从 TTIR 删除的特化参数仍作为普通 `i32` 打包进设备参数缓冲区，后续隐藏 grid 参数偏移与二进制约定不一致。
- **最终修改**：设备打包前将 `constants` 中顶层参数对应的 signature 标为 `constexpr`，复用现有 C++ 生成逻辑跳过设备参数；Python 侧仍保留完整参数个数检查和解析。Inductor 的非 2 次幂不特化规避已撤销，所有常量编译机会保留。
- **验证记录**：保留 `r0_numel=5120` 常量时，固定配置和正常自动调优分别为 0.08995/0.08953 ms，两个输出最大绝对误差均为 0.03125；运行时参数回归为 0.08974 ms。全新缓存模型模板仍含该常量，8 条 profiling 请求全部完成，平均 TTFT 14.504 s，prefill 窗口 25.964 s。最终官方非 profiling 复测输出吞吐 165.10 tokens/s、平均 TTFT 11.133 s、平均 TPOT 37.493 ms，16 条请求全部完成。
- **当前状态**：本地修复已通过算子、模型 profiling 和官方非 profiling 验证，尚未合入正式分支；正式合入前仍需覆盖多个特化标量、混合常量/运行时参数及普通 kernel 的 launcher 回归。
- **上游与发布状态**：上游 issue #2407 和 PR #2387 已确认同类问题；#2387 尚未合入。最新 main 经 #2685 shared launcher 重构已包含等价且支持嵌套参数路径的处理，但截至 2026-11-20，GitHub Release `v3.6.0.post2` 和 PyPI `3.2.0` 均不包含该修复。

### R-04：TE linearize 误改间接索引比较常量

- **现象**：Qwen3.5-0.8B 在当前 torch 2.13 / vLLM 0.33 环境中使用 TE Inductor，初始在 EngineCore profile run 报 `All tiling for [triton_unk_fused_index_2] are not runnable`；修复 inference downcast 后又暴露全 `!` 输出。
- **定位**：拆出失败 kernel 后，全 0 索引的 3072 个输出与 eager 全部不一致。`NPUTritonKernelOverrides.lt()` 入口打印证明原始常量为 `0`，最终生成文件却为 `2`，错误发生在 linearize 后处理。
- **根因**：reduction 维度替换用子串 `triton_helpers.` 匹配调用，同时误中 `npu_triton_helpers.lt(tmp0, 0)`，将 `0` 当作旧 reduction 维度并替换为新维度编号。
- **修复**：`codegen/triton.py` 在该替换条件中排除 `npu_triton_helpers.*`，保留上游 reduction helper 的原有处理。
- **引入与触发**：缺陷代码随 `8bfbb9ed1a` 于 2026-07-23 初始合入；`a94d4eb291` 于 2026-11-11 将第 0 维单张量 index 改走 TE kernel，Qwen3.5 的 RoPE 位置索引由此触发。旧 2.14 可用环境仍调用 ATen extern，不进入该路径。
- **验证**：最小 kernel 与 eager 一致；与 F-05、R-01 联合后，Qwen3.5-0.8B 冷编译、无插桩缓存复跑和最终复验均通过。单独保留 R-01 而撤掉本项及 F-05 修复时仍失败。
- **当前状态**：本地验证通过，尚未提交或推送；2026-11-21 获取的官方 `master@5eb53d43e5` 未包含修复。详见[专项定位](20261121_Qwen3.5-0.8B-TE-Inductor问题定位与修复.md)。

### O-22：Qwen3.6-35B-A3B-W8A8 旧 NZ 融合算子缺动态 shape 配置

- **现象**：A5 `214build` 环境中，35B-W8A8 在初始化 profiling 阶段报 `aclnnGroupedMatmulSwigluQuantWeightNZ` 的 `Config_Error(EZ1013)`，旧 NZ 算子的动态 shape JSON 配置不存在。eager、default 和 Inductor 共用该失败路径。
- **修复**：vLLM-Ascend `7db008b78` 将普通 W8A8 Dynamic MoE 融合分支切换到当前 CANN 已有动态 shape 配置的 `npu_grouped_matmul_swiglu_quant_v2`，同时适配权重/scale 列表、两个返回值并移除 V2 不支持的 `swiglu_limit`。融合功能保持开启。
- **验证**：`87fe251e7` 将接口 mock 替换为 `test_apply_gmm1_act_quant_small_moe`，以两个 token、两个专家在真实 NPU 上完成动态量化和 MoE 融合计算，结果为 `1 passed, 13 deselected`；35B eager 生成、固定输入、MMLU/GSM8K 和两轮性能流程均已执行。
- **边界**：该提交解决当前环境的启动阻塞，不代表固定输入 logprob 已满足容差，也不提供旧 NZ 与 V2 的有效性能对比。后续 GPQA 启动失败缺底层栈，需另行定位。
- **当前状态**：当前环境已修复。详见[专项定位](20261123_Qwen3.6-35B-W8A8-MoE融合算子V2启动修复.md)。

## 4. 临时规避或仅部分修复的问题

### W-01：动态 Norm+Quant 融合与 eager 非逐 token 一致

- **现象**：新增动态融合变体使命中数从 0 增至 55，但在 near-tie 处发生 argmax 翻转，无法通过逐 token 对拍。
- **定位**：eager 使用 `npu_add_rms_norm + npu_dynamic_quant` 两算子链，融合后改用 `npu_add_rms_norm_dynamic_quant`。两路存在 scale 计算/舍入顺序差异，55 个站点出现约 ±1～2 LSB 微差。
- **重要辨析**：CPU fp64 仲裁显示融合核反而更接近高精度真值，因此不能简单称为“融合核算错”；但项目验收要求与 eager 逐 token 一致，“更准”不能替代“一致”。确切分叉步骤仍未定位。
- **规避**：`b836b75d1` 将新动态变体放到 `fuse_norm_quant_dynamic` 下，默认 `False`，并把该开关纳入缓存 UUID。
- **当前状态**：默认路径恢复对拍，动态融合根因未修，属于临时规避。

### W-02：GDN 融合算子数值损坏

- **现象**：全零输入正常；常规随机输入在 T=64 时 absmax 约 9.1e7，T≥256 时输出 25%～62% NaN，`final_state` 可为 100% NaN。影响 eager、legacy 和 Inductor 所有路径。
- **定位**：问题指向 `torch_npu.npu_chunk_gated_delta_rule` 的 op-plugin/CANN 实现。旧可用性探针只用 zeros，无法触发数值路径。
- **规避**：`44c58bc72` 改用正常幅度的 q/k/v、sigmoid beta、负 decay，在 T=256 下检查 finite 和幅值；失败则禁用融合实现并回退 FLA。
- **验证记录**：受影响环境上探针返回 False，FLA 回退生效。
- **当前状态**：功能上有规避，底层算子未修；需在目标 CANN/A5 环境重新运行 `bug1_repro.py`，根治后才能评估撤盾。

### W-03：`FixFunctionalizationPass` 整体 no-op

- **现象**：恢复上游 Pass 会在 NPU 上崩溃；已确认其 RoPE target 引用 CUDA 专属 `torch.ops._C.rotary_embedding`。
- **定位**：直接触发点明确，但完整崩溃栈和非 RoPE 清理部分的受影响范围没有闭环，代码仍保留待定位注释。
- **规避**：`AscendPostGradPassManager.configure()` 用 `AscendNoopInductorPass` 替换整个 Pass，而不是只跳过 CUDA RoPE 部分。
- **风险**：当前 E2E 未暴露正确性问题，但非 RoPE 的 mutated-args/return 清理也不执行，属于升级和图卫生风险。
- **当前状态**：临时规避，需补最小崩溃栈并按算子域做最小禁用或 NPU 实现。

### W-04：QKNorm+RoPE 融合真图 0 命中

- **现象**：Inductor 轨 `qknorm_rope` 始终 0 命中；不融合时结果正确，因此这是优化缺失，不是数值 bug。
- **已定位环节**：TE 在第一次编译时才安装 NPU decomposition，晚于 configure 阶段的 pattern trace，pattern 锚点和真实 AOT 图形态不一致。
- **部分修复**：`44c58bc72` 在 pattern trace 前注册 decomposition，CPU 图实验 matched=1。
- **剩余问题**：真实 engine 仍 0 命中，pattern soup 与真图还存在 convert、view、weight mul 位次等形状差集。
- **当前状态**：部分修复。下一步应直接加载真实 FX dump 做逐节点匹配，不应重复修改已经验证过的注册时序。

### W-05：TE elide Pass 被 vLLM 的单槽替换语义遮蔽

- **现象**：TE 的 int→float→int 往返消除 Pass 在裸 `torch.compile` 路径可注入，但在 vLLM 编译路径不执行。
- **定位**：TE 将 Pass 安装到进程级 `post_grad_custom_post_pass` 单槽；vLLM 编译期通过线程级 `config.patch` 替换整个槽位，导致 TE 已安装的 compose 链被遮蔽。
- **规避**：`417fa2f7e` 增加 `AscendElideIntFloatIntPass`，在 VA PassManager 中显式调用 TE 私有函数，并将函数源码哈希纳入缓存 UUID；旧 wheel 不可用时告警跳过。
- **当前状态**：主轨优化已恢复，但 VA 仍耦合 TE 私有 API。根因修复应在 TE 注入机制，之后才能删除 wrapper。

## 5. 尚未解决的问题

### O-01：LoRA 混合请求 FULL 图挂死

- **现象**：`enable_lora=True` 引擎收到无 LoRA 请求时，在 FULL 图回放前的 `synchronize` 永久等待；PIECEWISE 和 LoRA-on 请求不挂。
- **定位进展**：曾怀疑 dummy run 覆写 `num_active_loras`，但修复该路径后仍挂 2 小时 37 分，相关改动未合入。下一候选是 GraphParams 仅以 `num_tokens` 为 key，同尺寸 LoRA/非 LoRA 图共享 events/handles。
- **当前状态**：未解决。建议优先验证将 key 扩为 `(num_tokens, has_lora)` 的假设，再恢复 Multi-LoRA/热加载 E2E；不能把 F-02 的 Meta 修复当作本问题修复。

### O-02：near-tie 数值分叉集合

- **涉及场景**：MoE 30B-A3B BF16/W8A8、8B 静态 W8A8 默认图模式、PCP 三层调试件，另有 PD transfer 同族记录。
- **现象**： greedy 对拍在首 token 或中间 token 分叉，部分续写均合理；当前若干用例仍为 strict xfail。
- **定位进展**：初步符合不同合法 kernel 的微小数值差在 logits 近似平票时翻转 argmax，但尚未用分歧点 logprob 间隙和高精度参考完成统一归因。
- **当前状态**：未闭环。需要区分“可接受数值噪声”与“真实数值缺陷”，再决定保留宽松判据还是修 kernel；不能长期只用 xfail 代替结论。

### O-03：FP8 block 原生权重 eager 加载失败

- **现象**：Qwen3-8B-FP8 eager 初始化时，`resolve_block_scales → aclnnInplaceCopy` 报 561103，Inductor 验证无法开始。
- **定位进展**：尚未区分 vllm-ascend 对该 checkpoint scale 布局的处理问题与 CANN 拷贝核缺陷。
- **当前状态**：未解决。按照“只要求 eager 能通过的场景”的约束，可从 Inductor 验收中跳过，但必须记录为平台/eager 缺陷，不能写成 Inductor FP8 已通过。

### O-04：MXFP8 eager `profile_run` 失败

- **现象**：Qwen3.8-27B-W8A8-MXFP8 TP2 eager 基线在 `npu_dynamic_mx_quant → aclnnDynamicMxQuantV2` 报 161001。
- **定位进展**：历史证据指向 CANN 动态 MX 量化核，尚未见修复记录。
- **当前状态**：外部阻塞、待新版本复测；同样不能计为 Inductor 量化验证通过。

### O-05：DSV4 W4A8 收尾段 device 尾错

- **现象**：历史冒烟中，非空生成、最终图模式和 TE marker 判据均已通过，但收尾段出现 QuantBatchMatmulV3 AICore 错误、HCCL fftsplus 1836 秒超时和 `npuSynchronizeDevice 507011`。
- **定位进展**：疑似量化 GEMM 核在图回放/teardown 边界出错，尚未定界。
- **当前状态**：未解决。需要独立复现收尾段，并区分权重/显存问题、图生命周期问题和 CANN kernel 问题。

### O-06：MoE TP4 并行预编译 `.ttir` 消失

- **现象**：冷编译约 90 秒时，4 个 rank 同时报 `.ttir` 文件不存在；目录残留 json 和 npubin，重试即过。
- **定位进展**：怀疑 `npu_triton_heuristics._compile_configs_batch` 的并行预编译与缓存目录清理发生竞态，静态分析未闭合。
- **当前状态**：低频未解决，暂靠重试。需要文件监视或复现窗口内日志证明删除者和使用者时序。

### O-07：legacy `fusion_pass` AOTAutogradCache 断言，修复已回退

- **现象**：torch 2.13 默认开启 AOT autograd cache 时，legacy `AscendCompiler → fusion_pass_compile` 返回 GraphModule，缓存层却要求 OutputCode，触发 `expected OutputCode, got GraphModuleImpl`。
- **历史修复**：`417fa2f7e` 在 legacy 编译期间关闭 `enable_autograd_cache` 和 remote cache，曾恢复与 eager 逐 token 一致。
- **回退**：`2f426bb16` 因“本分支只修改 Inductor，不修 legacy 轨”移除补丁；InductorAdaptor 自身有相同规避，不依赖该补丁。
- **当前状态**：legacy 问题回到社区存量状态。Stage 6 验证编排曾用 `TORCHINDUCTOR_AUTOGRAD_CACHE=0` 绕过，属于环境规避，不是当前产品代码修复。

### O-08：inproc legacy 轨 AscendConfig 双来源，修复已回退

- **现象**：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 且 identity cache 复用同一 VllmConfig 时，raw dict 已强制关闭 npugraph_ex/static kernel，单例仍保留旧 True，可能重新触发 #65 AOT cache 断言。
- **历史修复**：`146f0007c` 增加 step-7.5，将强制值同步到单例。
- **回退**：`ce8e97afd` 删除同步逻辑，理由是该缺陷仅属于 legacy 轨，不属于本次 Inductor 修改范围。
- **当前状态**：社区存量问题保留。需要在上游统一配置写入点或消除 identity-cache 双来源，而不是把历史提交误认为当前仍有修复。

### O-09：max-pool indices 数据契约错误

- **现象**：max_pool2d/3d_with_indices、backward 和 adaptive_max_pool2d 等编译用例失败，历史 AscendForge 黑名单中同族共 14 条。
- **定位**：op-plugin 的 `aclnnMaxPool2dWithMask` 返回 int8 bitmask，而 aten 契约要求 int64 argmax indices；torch_npu 因此将该族放入去分解路径，无法使用正常 Triton lowering。
- **当前状态**：根因已定位到 op-plugin，未见修复提交。该问题属于底层算子契约，不是 vllm-ascend 图切分问题。

### O-10：`aclnnSwiGluQuantV2` 动态模式越界

- **现象**：910B3/CANN 9.1.0-beta.1 上，`torch_npu.npu_swiglu_quant(..., quant_mode=1)` 触发 AIV 核 MTE DDR 越界；静态模式正常。
- **关联修复历史**：`3a11cfdbb` 和 `fc5c59b8f` 曾增加静态 SwiGlu+Quant 融合，但 `fe4fc22f0` 因现有 checkpoint 无目标链、零命中且无性能收益而完整回退。
- **当前状态**：F4 回退不等于底层动态算子已修复。该问题作为独立 CANN/torch_npu 缺陷候选保留，A5 可用性未验证。

### O-11：depyf 0.20.0 × torch 2.13 不兼容

- **现象**：设置 `VLLM_DEBUG_DUMP_PATH` 后，加载 Inductor kernel/wrapper 时抛 `patched_load_by_key_path() got an unexpected keyword argument 'set_sys_modules'`。
- **定位**：torch 2.13 的 codecache 新增 keyword-only `set_sys_modules`，depyf 仍使用旧四参补丁签名。
- **验证过的修复方向**：探针 shim 接受并转发该参数后，`PyCodeCache.load` 和 CPU torch.compile dump 均恢复；上游 issue 草稿已准备。
- **当前状态**：上游未闭环。当前不要使用 `VLLM_DEBUG_DUMP_PATH`，使用 `TORCH_COMPILE_DEBUG` 和 Inductor cache 产物。

### O-12：Mooncake NPU 扩展退出时堆损坏

- **现象**：`TransferEngine()` 进程内功能正常，但解释器退出时确定性出现 `corrupted size vs. prev_size`，exit 134，历史记录 4/4 复现。
- **定位**：指向第三方扩展 teardown/析构路径的堆损坏，尚未定位到具体实现。
- **当前状态**：外部阻塞。PD 测试曾按进程内结果验收并避开 returncode 断言，不能视为已修复。

### O-13：PD 示例没有真实 KV 传输

- **现象**：`offline_disaggregated_prefill_npu.py` 原样运行时 `kv_transfer_params=None`，send/receive 均为 0，D 侧本地重算 prefill。
- **定位**：MooncakeConnectorV1 是拉取式 P2P，示例缺 router 侧通过 `SamplingParams.extra_args` 注入 `kv_transfer_params` 的胶水。
- **验证记录**：补充胶水的探针曾观察到 P send=1、D receive=1，证明数据面可以实传。
- **当前状态**：示例缺陷未闭环，不是 Inductor 编译器本体问题。

### O-14：`compile_fx` 丢弃 decomposition 参数

- **现象**：断点地图记录 wrapper 未转发 `decompositions` 时会残留 `aten._to_copy`，并放大懒激活冲突。
- **当前代码核对**：`vllm_ascend/compilation/compiler_interface.py` 的 `compile_fx` 接收并递归传递 `decompositions`，但最终调用仍是 `aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)`；`fusion_pass_compile` 计算的 `select_decomp_table()` 没有进入 AOT。
- **状态辨析**：断点地图把 `aot_autograd(..., decompositions=select_decomp_table())` 写成修复方式，但当前产品代码并未落实，不能记为已修复。
- **影响范围**：该 wrapper 属于 legacy `AscendCompiler/fusion_pass` 路径，不是上游 InductorAdaptor 正门。
- **当前状态**：未解决，且需要修正断点地图的状态描述。

### O-15：TP2/TP4 EngineCore 出错后的父进程挂死

- **现象**：fusion_pass 断言触发后，TP1 能退出，TP2/TP4 只清理部分 worker，bench 父进程持续等待 RPC；历史记录中最长挂 12 小时。
- **定位**：初步指向多卡 EngineCore 在 `determine_available_memory` 等早期失败路径上的 worker 清理不完整。
- **规避**：Stage 6 脚本通过 `TORCHINDUCTOR_AUTOGRAD_CACHE=0` 避免触发 O-07，从而避免进入这条清理路径。
- **当前状态**：未解决。规避触发源不等于修复进程管理，需注入失败并验证所有 worker、RPC 和父进程都能退出。

### O-16：`vllm bench startup × fusion_pass × spawn` 独立崩溃

- **现象**：S2 启动 benchmark 报 `NoneType object has no attribute record_statistics_debug_info`，Inductor S3fp 路径正常。
- **定位**：仅归为 startup 工具、fusion_pass 与 spawn 子进程的组合问题，尚未深入定位对象生命周期。
- **当前状态**：未解决。Stage 6 用自有工具的 `init_s` 替代启动耗时口径，没有修复该工具链问题。

### O-17：AscendNPU-IR `hivmc-a5` 链接失败

- **现象**：链接时报 `isManualVFScope`、`setSanitizerAddrArgName` 等符号未定义。
- **定位**：声明存在，但链接库没有提供函数体，可能为 AscendNPU-IR CMake target 漏链或实现缺失。
- **影响边界**：当前 Triton 主链调用的是已经成功构建的 `bishengir-compile`，不调用 `hivmc-a5`，因此不阻塞既有 Inductor 冒烟。
- **当前状态**：上游构建缺陷未处理；若需要 HIVM/A5 工具必须单独修复，不能将其与 vllm-ascend 构建混为一谈。

### O-21：Qwen3.8-27B-W8A8 三条执行路径的数值一致性差异

- **现象**：eager 与原始 Inductor 的固定输入 token/text 均为 0/8 一致，logprob 平均绝对差 0.338180、最大差 2.888581；eager 与 default 也为 0/8，default 与原始 Inductor 同样为 0/8。相同任务样本下，Inductor 的 MMLU 89.0%→91.0%、GSM8K 94.5%→95.5%，分数没有下降。
- **实现差异**：default 与 Inductor 共用 CANN `npu_add_rms_norm_dynamic_quant`，动态量化、量化矩阵乘和 SwiGLU 也保留原生调用；差异主要来自 Inductor 生成的 sigmoid 融合 kernel 和独立 RMSNorm Triton kernel。
- **对照结果**：关闭图捕获不改变 Inductor 输出。以 default 为参考，只补 sigmoid 中间舍入时为 4/8 一致，再结合原生 RMSNorm 诊断时为 8/8 且 logprob 差异为 0；单独保留原生 RMSNorm 或取消 NormQuant 融合均不能使 eager 与 Inductor 恢复一致。
- **性质与边界**：这些结果首先证明数值实现不同，不证明 Inductor 的数学精度一定更差。CANN NormQuant 融合用于解释 eager 与编译路径的差异，不是 default 与 Inductor 共享路径上的差异；其在 eager/default 间是否为唯一来源尚未证明。
- **当前状态**：未解决。sigmoid 原型因全局作用于 TE BF16/FP16 sigmoid 而已撤回，原生 RMSNorm 和取消融合仅为诊断，不作为最终修复。详见[专项定位](20261122_Qwen3.8-27B-W8A8精度定位.md)。

## 6. 工程与部署问题

### E-01：DSV4 权重名实不符及 TP4 加载 OOM

- **现象**：名为 `DeepSeek-V4-Flash-w4a8-mtp` 的权重，其实际量化描述是 W8A8_DYNAMIC；TP4 加载时单卡已分配约 60.2G，物理容量约 61G，权重阶段即 OOM。
- **定位**：权重命名与实际格式不一致，加上量化路径的分片、复制张量和 scratch 内存，导致 64G 卡无法容纳。
- **当前状态**：未解决，受正确权重和硬件资源约束；与 O-05 的生成收尾段尾错是两个独立问题。

### E-02：A5 环境缺 cp312 vllm_ascend 预编译 wheel

- **现象**：原预编译 wheel 直装步骤没有可用的 cp312 包。
- **处理**：部署时改为 editable 现场编译，首轮约 50 分钟。
- **当前状态**：当时已规避；若要求可重复部署，仍需确认并发布匹配的 cp312 构建产物。

## 7. 待复测或尚未形成完整证据链的问题

### R-02：旧 triton-ascend 的 JIT monitor 导入失败

- **现象**：triton-ascend 3.2.2 的 `triton.knobs` 导入因 libtriton 符号问题失败，上游 JIT monitor `activate()` 随之失败。
- **规避**：`f4bce8cd1` 对 monitor 激活做异常捕获，告警后继续运行；Inductor lazy init 不受影响。
- **当前状态**：旧栈仅有降级，不是根因修复。A5 已使用 3.6.0.post2，需要重新执行 `from triton import knobs` 和 monitor 激活验证，再决定保留还是删除该遗留项。

### U-01：不同后端复用编译缓存 hash

- **已有信息**：会议记录只写明“有一个 hash 问题，导致不同后端复用 hash”。
- **已有相关加固**：部分融合 Pass 已加入配置 UUID，torch_npu elide Pass 也将其函数源码哈希纳入 UUID，但这些不能证明整个缓存 key 已完整隔离后端。
- **缺失证据**：没有独立最小复现、错误缓存目录/键、涉及的两个后端及配置、错误加载日志和根因提交。
- **当前状态**：线索待确认。下一步应固定同一模型和输入，交替使用 legacy/Inductor 后端并共享 `VLLM_CACHE_ROOT`，检查是否发生跨后端加载，再决定修复点。不能将该线索写成已确认的根因或已修复问题。

## 8. 测试代码和测试环境自身的缺陷

这些问题不会直接破坏线上运行，但会使验证结论失真，应与产品 bug 分开处理。

### T-01：8B W8A8 默认图模式测试没有启用 Inductor

- **位置**：`tests/e2e/pull_request/one_card/inductor/test_inductor_track_w8a8.py` 中 `test_w8a8_8b_inductor_track_default_cg_gate` 和对应 eager parity 用例。
- **问题**：kwargs 没有 `compilation_config.backend="inductor"`，但断言却检查 Inductor cache marker 或将用例命名为 Inductor 对拍。
- **影响**：测试不能证明所声明的 Inductor 路径；marker 还可能受到同进程/共享缓存中其他用例影响。
- **状态**：当前源码仍可确认，需补正确 backend 并使用隔离缓存后重跑。

### T-02：MoE W8A8 TP4 配置字段放置错误

- **位置**：`tests/e2e/pull_request/four_card/inductor/test_inductor_track_moe_w8a8_tp4.py`。
- **问题**：`_TRACK` 将 `weight_nz_mode` 放入 `additional_config["ascend_compilation_config"]`；该嵌套配置没有此字段且禁止未知键，正确字段位于外层 AscendConfig。
- **影响**：用例可能因配置校验失败，或者不能按声明的 NZ 配置运行。
- **状态**：当前源码仍可确认，需改正配置位置并重跑，而不能沿用历史通过记录。

### T-03：部分“冷编译”测试不清空固定缓存目录

- **位置示例**：`test_inductor_track_long16k.py`、FP8/MXFP8/W4A8 的部分用例。
- **问题**：只执行 `mkdir(parents=True, exist_ok=True)`，未保证目录为空。
- **影响**：第二次运行可能直接命中缓存，日志中的 cold init 和编译覆盖不再可信。
- **状态**：当前源码仍可确认。应使用独立临时目录或显式清理，并断言冷编译日志/产物确实新增。

### T-04：`mark_dynamic` 标记污染后续测试

- **问题**：`torch._dynamo.mark_dynamic` 是 tensor 上的粘滞属性，复用输入会把前一 case 的动态标注带入后一 case。
- **现象**：后续 case 报 `ConstraintViolationError`，表面错误指向当前 case，实际污染源在前一 case。
- **处理**：每个 case 创建全新输入 tensor，不能仅靠清理全局配置。

### T-05：Qwen3-Next 探针显存配置错误

- **问题**：探针未设置 `gpu_memory_utilization`，默认 0.9 在 KV 预留阶段 OOM；benchmark 显式使用 0.80 则能运行。
- **处理**：大模型探针统一显式设置不超过 0.80，避免把配置问题误判为模型或 Inductor 缺陷。

### T-06：缓存检查文档路径缺少 rank 层级

- **问题**：旧检查命令搜索 `<hash>/inductor_cache/`，实际路径为 `<hash>/rank_0_0/inductor_cache/`，因而错误得到零 marker。
- **处理**：按包含 rank 目录的真实路径检查；正确路径下历史记录找到 10 个 TE 标记文件和 98 个 `.npubin`。

### T-07：测试环境缺 `pytest_mock`

- **问题**：`tests/ut/ops/test_comm_utils.py` 导入 `pytest_mock` 失败，导致全量测试收集中断。
- **处理**：安装 `pytest-mock` 并纳入环境依赖，不应长期通过 `--ignore` 跳过整个文件。

## 9. 建议的后续闭环顺序

1. **先修验证可信度**：处理 T-01～T-07，统一缓存隔离和 eager 基线，再判断 O-02、O-03、O-04 的真实状态；R-01、R-03 已有本地验证方案，后续重点是合入评审和针对性回归，不再重复列为待复测项。
2. **优先处理明确影响运行的缺陷**：O-01 LoRA 挂死、O-03/O-04 量化 eager 崩溃、O-05 W4A8 尾错、O-06 编译竞态、O-15 多卡清理挂死。
3. **修正 legacy 编译问题**：O-14 先补 decomposition 转发和回归，再处理 O-07/O-08；这些修复应独立上游化，不与 Inductor 正门混淆。
4. **确认缓存正确性**：为 U-01 补最小复现，同时检查后端、Pass 配置、torch_npu/TE 版本和自定义 Pass 源码是否都进入缓存 key。
5. **清理临时规避**：在目标 CANN/A5 版本复测 W-01、W-02、W-03、W-05、R-02，根因消失后再撤规避，不能仅凭版本升级直接删除防护。
6. **推动外部问题闭环**：O-09～O-12、O-17 需要 op-plugin/CANN、depyf、Mooncake、AscendNPU-IR 侧修复或版本验证，并保留最小复现。

## 10. 主要证据入口

- [功能遗留总表](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/04_总表.md>)
- [torch_npu / TE 遗留明细](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/01_torch章.md>)
- [vllm-ascend 遗留明细](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/02_vllm章.md>)
- [第三方与工程问题明细](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>)
- [GDN bug1 台账与最小复现说明](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/torch-npu bug/bug1_npu_chunk_gated_delta_rule_数值破坏.md>)
- [推理态 `_version` bug2 台账](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/torch-npu bug/bug2_memoized_downcast_推理态.md>)
- [Inductor 断点与调试地图](<../../../inductor-meta-worktree/vllm-ascend/docs/source/developer_guide/performance_and_debug/inductor_track_breakpoint_map.md>)
- [Stage 5 遗留清单](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/TODO.md>)
- [Stage 6 性能与复测记录](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>)
- [A5 部署问题与遗留](../a5-25-cann920-env-deploy.md)
- [当前适配、测试和环境状态](../vllm-ascend-Inductor适配现状.md)
- [1105 会议中的缓存 hash 线索](../../手工文档/1105会议.md)
- [功能缺失、性能优化和覆盖缺口总结](../feature/Inductor适配功能性能与覆盖缺口.md)

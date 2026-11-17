# vllm-ascend Inductor 适配测试工作台账

## 1. 记录口径

本文按**测试项**逐行记录：每行同时给出测试环境、模型与运行参数、输入、测试方法、结果和证据来源。相同模型在不同图模式、TP、量化格式或输入下分别记录，不再把环境、参数、输入和方法拆散描述。

- 本文没有重新运行测试，结果来自历史验证文档和当前测试代码。
- “通过”仅对该行所列环境和判据有效；“冒烟”不等于完整精度或性能通过；xfail、skip、配置待修单独标明。
- 原文没有记录的字段明确写“未记录”，不做推测。
- 所有来源均可点击，测试代码链接附行号，便于直接定位。

## 2. Stage 1～3：主链路、图模式和量化验收

| 测试项 | 测试环境 | 模型与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|---|
| 首次 Inductor 主链路 | 910B2；vLLM/vllm-ascend 0.27.1；torch/torch_npu 2.13；triton-ascend 3.2.x | Qwen3-0.6B BF16；`backend=inductor`；初期 `cudagraph_mode=NONE`；`max_model_len=4096`、`max_num_seqs=16` | 两条：`The capital of France is`、`给我一句话介绍长城。`；greedy 48 tokens；另有 `temperature=0.7、top_p=0.8、seed=10086` 同种子采样 | 与 eager 同输入逐 token 对拍；检查 InductorAdaptor、`output_code.py`、`npu_triton_heuristics` 和 Triton kernel | 输出一致；10 个 TE marker 文件；冷编约 467.6s；证明 compile-only 主链路可用。[Stage1 T2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage1/_notes/t2_evidence/RESULTS.md>)、[smoke 脚本](<../../../inductor-meta-worktree/docs/vllm/env/scripts/smoke_qwen3_inductor_track.py>) |
| PIECEWISE 捕获与回放 | 910B2；同 Stage1 软件体系 | Qwen3-0.6B；Inductor + PIECEWISE；`max_model_len=4096`、`max_num_seqs=16`；spawn | 同上中英文 prompt；greedy | NPUGraph init/replay spy、compilation counter、eager 对拍、TE 产物检查 | 历史计数 capture 145=29×5、replay 464=29×16；输出与 eager 一致。[Stage2 验收](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage2/04_测试与验收.md>) |
| TP2 PIECEWISE | 910B2×2；spawn；设备 ID 升序 | Qwen3-0.6B；TP=2；PIECEWISE；`max_model_len=4096`、`max_num_seqs=16` | 同两条中英文 prompt；`temperature=0.7、top_p=0.8、seed=10086`、48 tokens | 与 eager TP2 比较完整输出文本/token，检查双 rank 缓存目录 | 输出与 eager TP2 一致；支持每 rank 独立编译缓存。[TP2 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_tp2.py>) |
| 在线 serve | 910B2；spawn | Qwen3-0.6B；Inductor + PIECEWISE；其余与离线 smoke 对齐 | 与离线轨相同 prompt 和 greedy 参数 | 启动 OpenAI server，请求 `/v1/completions`，与离线 Inductor 输出比较，并检查服务端编译和捕获日志 | 在线输出与离线轨一致。[Stage2 T5](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage2/04_测试与验收.md>) |
| FULL 族及默认图模式 | 910B2；进程内 `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Qwen3-0.6B；`FULL_AND_PIECEWISE/FULL/FULL_DECODE_ONLY/DEFAULT`；`max_model_len=4096`、`max_num_seqs=16`、gmu 0.55 | 两条中英文 prompt；greedy 16 tokens | 按 runtime mode 和 batch size 分组统计 NPUGraph init/replay，闭合分段与全图公式，并检查 final_cg | F&P 历史计数 150=29×5+5；FULL/FDO 每个尺寸一张整图、无内层 piecewise 图；默认 -O2 落到 F&P。[FULL 图测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_full_graphs.py>) |
| 动态 W8A8 的 NZ 数值矩阵 | 910B2；msmodelslim 生成量化权重 | Qwen3-0.6B W8A8_DYNAMIC；`quantization=ascend`；PIECEWISE；`max_model_len=1024`；`weight_nz_mode=0/1` | 共用 `PROMPTS_SHORT` 四条英文 prompt | `compare_logprobs`：prefill token 一致、logprob 容差 0.0689；decode 对 top-k 交叉查值 | NZ 0/1 均有历史通过记录；这是数值对拍，不只是非空生成。[W8A8 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_w8a8.py>)、[数值工具:365](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/utils.py#L365>) |
| 静态 W8A8 融合命中 | 910B2；进程内、独立冷缓存 | Qwen3-8B-W8A8 静态件；PIECEWISE；`max_model_len=1024`、`max_num_seqs=4`、gmu 0.85、NZ=0 | `The capital of France is`；greedy 8 tokens | 冷编后读取 `VllmPatternMatcherPass.match_table`，并检查非空生成 | `rmsnorm_quant_fusion_pass` 历史命中 3，证明静态融合真实执行。[静态融合测试:163](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_w8a8.py#L163>) |
| 动态量化 pattern 修复与数值仲裁 | 910B2；进程内、独立冷缓存；另有 CPU fp64 仲裁 | Qwen3-0.6B W8A8_DYNAMIC；上行参数基础上开启 `fuse_norm_quant_dynamic=True` | 同一组 `PROMPTS_SHORT`；greedy 8 tokens；算子级比较融合核、eager 两算子链和 CPU fp64 | 检查 bias=None/view pattern 命中、eager 逐 token 差异和高精度误差 | 双轨历史命中 55；融合核更接近 fp64，但与 eager 非逐 token 一致，因此最终默认关闭该动态融合，而不是默认开启。[W8A8 测试:117](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_w8a8.py#L117>)、[bug W-01](../bugfix/Inductor适配Bug定位与修复追踪.md) |

## 3. Stage 4：模型、执行层和分布式功能验收

| 测试项 | 测试环境 | 模型与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|---|
| Qwen 默认图模式 × TP2 | 910B2×2；spawn | Qwen3-0.6B；TP2；显式 F&P 和不设 cg 的默认 -O2 两组；`max_model_len=4096`、`max_num_seqs=16`、gmu 0.55 | 两条中英文 prompt；greedy 48 tokens | 分别检查非空生成和默认档 final_cg；每组独立缓存目录 | 显式 F&P 与默认 -O2→F&P 均有通过记录。[TP2 cg 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_tp2_cg.py>) |
| MRv2 执行层 | 910B2；`VLLM_USE_V2_MODEL_RUNNER=1`；进程内 | Qwen3-0.6B；`max_model_len=4096`、`max_num_seqs=8`、gmu 0.4；Inductor 默认图档 | `PROMPTS_SHORT`；greedy 48 tokens | 检查实际 runner 类型、编译/捕获计数增量，并与 v2 eager 对拍 | 探针基线 29 次编译、120 次捕获；当前断言计数大于 0 和 eager 一致。[MRv2 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_mrv2.py>) |
| DeepSeek MLA 单卡 | 910B2；spawn | DeepSeek-V2-Lite BF16；PIECEWISE；`max_model_len=4096`、`max_num_seqs=8`、gmu 0.85 | 两条中英文 prompt；greedy 48 tokens | 与单卡 eager 逐 token 对拍，检查 final_cg 和 TE marker | 2×48 tokens 一致，PIECEWISE 和 TE 产物均在。[MLA 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_mla.py>) |
| DeepSeek MLA TP2 | 910B2×2；spawn | DeepSeek-V2-Lite BF16；TP2；其余与单卡相同 | 同一组中英文 prompt；greedy 48 tokens | 与 eager TP2 逐 token 对拍，检查 final_cg 和 TE marker | 历史高级档 gate 通过。[MLA TP2 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_mla_tp2.py>) |
| DeepSeek MLA 量化 | 910B2；spawn | DeepSeek-V2-Lite-W8A8_DYNAMIC；单卡 PIECEWISE；`max_model_len=4096`、`max_num_seqs=8`、gmu 0.85；quantization=ascend | 同一组中英文 prompt；greedy 48 tokens | eager 逐 token 对拍；检查 TE、MoE、动态量化产物和 match_table | 2×48 tokens 一致；产物计数仅观测，match_table 不强制大于 0。[MLA W8A8 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_mla_w8a8.py>) |
| MoE BF16 TP4 | 910B2×4；spawn | Qwen3-30B-A3B BF16；TP4；默认 -O2→F&P；`max_model_len=4096`、`max_num_seqs=16`、gmu 0.80 | 两条中英文 prompt；greedy 48 tokens | 运行 gate、final_cg、TE marker、MoE/extern 产物统计；另做 eager 对拍 | gate 通过；eager 对拍存在 near-tie 分歧并保留 xfail，不能算严格数值全过。[MoE 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/four_card/inductor/test_inductor_track_moe_tp4.py>) |
| MoE W8A8 TP4 | 910B2×4；spawn | Qwen3-30B-A3B-W8A8；TP4；默认 F&P；`max_model_len=4096`、`max_num_seqs=16`、gmu 0.80；声明 NZ=1 | 两条中英文 prompt；greedy 48 tokens | 运行 gate 和 eager 对拍，统计量化产物 | 历史 gate 有记录、对拍 xfail；但当前 `_TRACK` 把 NZ 错放进 `ascend_compilation_config`，需修正后重跑。[测试:65](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/four_card/inductor/test_inductor_track_moe_w8a8_tp4.py#L65>)、[bug T-02](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| Dense 32B BF16 | 910B2×2；spawn | Qwen3-32B BF16；TP2；默认 F&P；`max_model_len=8192`、`max_num_seqs=16`、gmu 0.85 | 两条中英文 prompt；greedy 48 tokens | 与 eager TP2 对拍，检查 final_cg 和 TE marker | 两个 prompt 均 48/48 一致。[Dense32B 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_dense32b_tp2.py>) |
| Dense 32B W8A8 tuned | 910B2×2；spawn | Qwen3-32B_w8a8_tuned；TP2；默认 F&P；`max_model_len=8192`、`max_num_seqs=16`、gmu 0.85；quantization=ascend | 两条中英文 prompt；greedy 48 tokens | 与 eager 对拍并检查 TE marker | 历史记录逐 token 一致。[Dense32B W8A8 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_dense32b_w8a8_tp2.py>) |
| 16K 长输入 | 910B2；spawn | Qwen3-8B BF16；默认 F&P；`max_model_len=16384`、`max_num_seqs=4`、gmu 0.80 | 固定 `random.Random("stage4-me_16k")` 的 16352 个 token ID；greedy 32 tokens、`ignore_eos=True` | 同 token ID 对 eager/Inductor 逐 token 对拍，检查 final_cg、TE 产物和冷启动耗时 | 历史结果 32/32 一致；冷 init 551s，单编译 range 约 402s。当前固定缓存目录未清空，需用独立空目录重跑冷编。[long16k 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_long16k.py>)、[bug T-03](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| 8B 静态 W8A8 默认图档 | 910B2；测试代码当前配置待修 | Qwen3-8B-W8A8；声明默认 -O2→F&P；`max_model_len=1024`、`max_num_seqs=4`、gmu 0.85、NZ=0 | `PROMPTS_SHORT`；gate 非空生成；另有 eager 对拍 | 检查 final_cg、TE marker 和 token/logprob | 历史 gate 通过、对拍 strict xfail；但当前两个用例都未传 `backend="inductor"`，必须补配置和隔离缓存后重跑。[测试:208](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_w8a8.py#L208>)、[bug T-01](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| Eagle3 推测解码 | 910B2；spawn | Qwen3-8B + Qwen3-8B-speculator.eagle3；`num_speculative_tokens=3`；FDO；capture `[12]`；`max_model_len=2048`、`max_num_seqs=8`、gmu 0.7 | `The capital of France is`、`Hello, my name is Tom, I am`；greedy 32 tokens | spec-on eager 与 spec-on Inductor 逐 token 对拍，检查 final_cg 和 TE marker | 历史记录通过。[Spec 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_spec.py>) |
| DFlash 推测解码 | 910B2；spawn | Qwen3-8B + Qwen3-8B-DFlash-b16；其余与 Eagle3 相同 | 与 Eagle3 相同 | 与 Eagle3 相同 | 历史记录通过。[Spec 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_spec.py>) |
| ngram / ngram_gpu | 910B2；spawn | Qwen3-0.6B；ngram lookup 3～5、spec tokens 3；`max_model_len=1024`、`max_num_seqs=8`、gmu 0.6、capture `[1,2,4,8]`；默认 F&P | 两条英文 prompt；greedy 32 tokens | spec-on eager 与 spec-on Inductor 对拍；配置重构后另验证 ngram_gpu 和 legacy 双路径 | ngram 与后续 ngram_gpu 路径均有通过记录。[Spec 测试:65](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_spec.py#L65>) |
| suffix | 910B2；依赖不满足时 skip | Qwen3-0.6B；spec tokens 10、suffix factor 2.0、cached requests 1000；`max_model_len=1024`、`max_num_seqs=8`、gmu 0.6、max capture 16 | 两条英文 prompt；greedy 32 tokens | 仅在 arctic-inference 可用时执行同样的 eager 对拍 | 依赖安装会破坏 torch wheel 组合，因此跳过；没有通过结论。[Spec 测试:73](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_spec.py#L73>) |
| PCP2 冒烟 | 910B2×2；V2 runner | DeepSeek-V3.2-W8A8-Pruning 三层调试件；TP1×PCP2；FDO capture `[4]`；`max_model_len=1024`、`max_num_batched_tokens=1024`、`max_num_seqs=4`；chunked prefill、prefix cache、cp interleave/block 128、gmu 0.8 | 两条英文 prompt；greedy 32 tokens | 检查 PCP2 下编译、生成、final_cg 和 TE 产物；输出只记录，不作通用精度基准 | PCP 激活和冒烟通过；三层调试件输出退化，只作记录。[PCP 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_pcp_smoke.py>) |
| FP8 block | 910B2；eager 基线阻塞 | Qwen3-8B-FP8，block `[128,128]`；单卡；`max_model_len=2048`、`max_num_seqs=8`、gmu 0.85 | 两条英文 prompt；greedy 32 tokens | 先跑 eager，再跑 Inductor 并做逐 token 对拍 | eager 初始化即报 `aclnnInplaceCopy 561103`，未进入有效 Inductor 对拍。[FP8 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_quant_fp8.py>)、[bug O-03](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| MXFP8 | 910B2×2；eager 基线阻塞 | Qwen3.8-27B-W8A8-MXFP8；TP2；`max_model_len=4096`、`max_num_seqs=8`、gmu 0.85；quantization=ascend | 两条英文 prompt；greedy 32 tokens | 先跑 eager，再跑 Inductor 并做逐 token 对拍 | eager `profile_run` 报 `aclnnDynamicMxQuantV2 161001`，未形成 Inductor 验证。[MXFP8 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/two_card/inductor/test_inductor_track_quant_mxfp8_tp2.py>)、[bug O-04](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| W4A8 DSV4 | 910B2×4；后续 Stage6 环境另发现权重/显存问题 | DeepSeek-V4-Flash-w4a8-mtp；TP4；`max_model_len=8192`、`max_num_seqs=8`、gmu 0.85；`VLLM_USE_BREAKABLE_CUDAGRAPH=0` | 两条中英文 prompt；greedy 32 tokens | 只测启动、非空生成、final_cg 和 TE marker，不做第二个 eager 引擎对拍 | 历史冒烟判据曾过，但收尾段有 AICore/HCCL 尾错；后续发现权重实体为 W8A8_DYNAMIC 且 TP4 加载 OOM，不能算完整 W4A8 通过。[W4A8 测试](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/four_card/inductor/test_inductor_track_quant_w4a8_dsv4_tp4.py>)、[bug O-05/E-01](../bugfix/Inductor适配Bug定位与修复追踪.md) |

## 4. 专项与最小复现

| 测试项 | 测试环境 | 模型/参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|---|
| LoRA shrink Meta | 910B2；Python Meta/AOT 单测 | LoRA `bgmv_shrink/sgmv_shrink`；在既有 capability gate 下测试 | Meta tensor、FakeTensor、`aot_eager` 编译链 | 对比补 Meta 前后的 FakeTensor 传播和完整 LoRA trace | 补 Meta 后链路通过，历史记录 68 个 LoRA UT 通过；不代表运行时 LoRA 全场景通过。[bug F-02](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| LoRA FULL 挂死 | 910B2；FULL/F&P 与 PIECEWISE 对照 | Qwen3.5-4B + SQL adapter；`enable_lora=True` | 无 LoRA 请求与 LoRA-on 请求对照 | 比较 FULL 回放和 PIECEWISE；曾试修 `num_active_loras` | 无 LoRA 请求在 FULL 回放前挂死，试修后仍挂 2h37m；Multi-LoRA/热加载 E2E 未完成。[bug O-01](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| PD 数据面 | 910B2；双进程 | MooncakeConnectorV1；原示例与补 `kv_transfer_params` 探针对照 | greedy 请求；具体模型在汇总文档未记录 | 统计 P send、D receive 和 D 侧是否重算 prefill | 原示例 send/receive=0；补胶水后 P send=1、D receive=1。仅冒烟级。[bug O-13](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| RoPE fake 别名 | 910B2；engine + 最小 twin-op | Eagle3；RoPE fake 直接返回输入 vs `empty_like` | draft head 下游 attention view/reshape；最小 fake-tensor 复现 | engine 复现、output_code diff、twin-op 三层对照 | 证实错误别名导致 `setStorage` 越界；改 `empty_like` 后通过，acceptance 在基线范围。[bug F-01](../bugfix/Inductor适配Bug定位与修复追踪.md) |
| GDN 数值 | 910B2；op-plugin/CANN 最小复现 | `torch_npu.npu_chunk_gated_delta_rule`；T=64/256 | 全零对照；正常随机 q/k/v、sigmoid beta、负 decay | 统计 absmax、NaN 比例和 final_state；再验证 VA 探针回退 | 全零正常，T=64 absmax 约 9.1e7，T≥256 大量 NaN；探针盾回退 FLA，底层算子未修。[GDN bug1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/torch-npu bug/bug1_npu_chunk_gated_delta_rule_数值破坏.md>) |
| gather 融合切断微基准 | 910B2；compiled vs eager | index_select/slice+cat/argmax；`8192×4096 fp16` | 固定微基准张量 | 比较 pure/cut 链耗时，并检查 output_code 是否 extern | 整链已是单 Triton kernel、零 extern；cut 链仍慢 1.26～1.68×，定位为 kernel 质量。[CHECKPOINT A7](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/CHECKPOINT.md>) |
| AscendForge 黑名单回归 | 910B2；上游 torch.compile 迁移用例 | 113 个黑名单用例；75 个可解析，38 个本地语料缺失 | 各上游 UT 自带输入 | 映射节点 ID、collect 交叉核对、转发 rerun 并汇总 | 14 passed/59 failed；14 条旧黑名单转绿，其余用于逐条归因，不是全量通过。[CHECKPOINT A5](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/CHECKPOINT.md>) |
| depyf × torch 2.13 | CPU，不依赖 NPU 或模型 | depyf 0.20.0；设置 `VLLM_DEBUG_DUMP_PATH`；对比接受 `set_sys_modules` 的 shim | 最小 `PyCodeCache.load` 和 CPU torch.compile | 复现签名错误，再用 shim 验证修复方向 | 旧四参签名导致 TypeError；shim 后 CPU dump 恢复。当前仍避免 `VLLM_DEBUG_DUMP_PATH`。[PI-3P-01](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/function design pending issue/03_第三方与工程章.md>) |

## 5. 性能测试：每个模型对应的环境、参数、输入和方法

### 5.1 Stage3 后 Qwen 性能摸底

| 模型 | 测试环境与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|
| Qwen3-0.6B BF16 | 910B2 单卡；s1/s2/s2p/s3/s3b/s3fp 六场景；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache、捕获档 `[1,2,4,8,16,24,32]` | 五组：128/512/b1；1024/256/b1；1024/256/b8；1024/256/b32；4096/128/b8。随机 token 按负载 ID 固定 | 每场景 warmup 1 次后 3 次取中位数；测 TPOT、吞吐和 TTFT | 当时最好档相对生产默认 1.35～1.83×；该基线后续已失效，不能作为当前结果。[perf-after-stage3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/perf-after-stage3/RESULTS.md>) |
| Qwen3-8B BF16 | 910B2；TP1 六场景，TP2 仅 eager/生产默认/Inductor 分段三场景；其余同上 | 与上行完全相同的五组负载 | 与上行相同；TP2 单独做双卡对照 | 当时最好档约 1.00～1.08×，基本追平；不能外推为 A5 结果。[perf-after-stage3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/perf-after-stage3/RESULTS.md>) |
| Qwen3-8B-W8A8 静态件 | 910B2 单卡；quantization=ascend、NZ=0；六场景；其余同上 | 与上行相同的五组负载 | 与上行相同 | 当时 decode 0.75×、prefill 1.39×；后续 Stage5/6 同口径复测已追平，旧 0.75× 不再是当前结论。[perf-after-stage3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/perf-after-stage3/RESULTS.md>)、[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |

### 5.2 Stage5 专项 A/B

| 测试项 | 测试环境与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|
| custom_ops 全矩阵 | Stage5 历史 910B 环境，RESULTS 未统一注明具体卡型；Qwen3-0.6B、Qwen3-8B、Qwen3-8B-W8A8；`all/none/mixed` 三臂，Inductor 混合档 | decode、b8、b32、prefill 四类负载；同负载跨臂一致 | 3 模型×4 负载×3 次重复，比较 tok/s 中位数 | `all` 最终最优；none 在 8B/W8A8 回退约 4%～12%，mixed 无额外挽回，因此未设置时默认 all。[Stage5 M2](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| max_autotune | 同一 Stage5 环境；Qwen3-0.6B；比较整档 autotune 与基线 | decode、b32、prefill 等，另记录冷编时间 | A/B 比较性能和编译耗时 | 性能无可靠收益；冷编约 770.6s→1528.8s，默认不开启，显式配置尊重用户。[Stage5 M3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| async × Inductor | 同一 Stage5 环境；Qwen3-0.6B；async on/off | offline decode、b8 等 | 同配置开关 async，检查运行和吞吐 | 可运行但 offline 无收益，decode/b8 约下降 1.6%～4.3%，保持默认关闭。[Stage5 M5](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| comprehensive padding | 同一 Stage5 环境；Qwen3-0.6B；padding 0/1 | decode、b8、b32、prefill；3 次重复快档 | A/B 比较 | 各负载差异在噪声内，保持上游默认 True。[Stage5 M5](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| SwiGlu+Quant F4 | 910B3/CANN 9.1.0-beta.1 等 Stage5 环境；Qwen3-8B-W8A8；静态/动态算子语义分别探测 | 静态融合链、动态 `quant_mode=1`；on/off E2E A/B | 检查目标图链、融合命中、算子语义和端到端性能 | 当前 checkpoint 的 down_proj 是 BF16，目标链不存在、零命中；on/off 差异无意义，最终整体回退。动态模式另有 AIV/MTE 越界。[Stage5 M4](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |

### 5.3 Stage6 全模型性能矩阵

表中结果比值依次为 **decode / b1 / b8 / b32 / prefill**。每行都重复列出实际环境、输入和方法，不依赖其他行补全。

| 模型 | 测试环境与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|
| Qwen3-0.6B | 910B3 64G；TP1；BF16；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | `128/512/b1`；`1024/256/b1`；`1024/256/b8`；`1024/256/b32`；`4096/128/b8`；按负载 ID 固定随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **0.98 / 0.98 / 0.98 / 1.03 / 1.09**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3-8B-W8A8 | 910B3 64G；TP1；quantization=ascend、NZ=0；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表收紧随机 token 上限 | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.01 / 1.02 / 1.02 / 1.00 / 1.01**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| DeepSeek-V2-Lite-W8A8 | 910B3 64G；TP1；quantization=ascend、NZ=0；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按 DeepSeek 词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.01 / 1.00 / 1.00 / 1.01 / 1.00**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| DeepSeek-V2-Lite BF16 | 910B3 64G；TP1；BF16；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按 DeepSeek 词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **0.93 / 0.98 / 1.00 / 1.01 / 0.97**；decode 为明确短板。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3-32B-W8A8-QuaRot | 910B3 64G；TP1；quantization=ascend、NZ=0；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.00 / 1.01 / 1.00 / 1.00 / 1.01**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3-30B-A3B-W8A8 | 910B3 64G；TP1；quantization=ascend、NZ=0；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.01 / 1.02 / 1.02 / 1.01 / 1.02**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3-30B-A3B BF16 | 910B3 64G×2；TP2；BF16；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.02 / 1.01 / 1.05 / 1.05 / 1.03**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3-32B BF16 | 910B3 64G×2；TP2；BF16；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.00 / 1.01 / 1.03 / 1.01 / 1.01**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3.5-35B-A3B-W8A8-MTP | 910B3 64G×2；TP2；quantization=ascend、NZ=0；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.00 / 1.01 / 1.06 / 1.14 / 1.07**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |
| Qwen3-Next-80B-A3B BF16 | 910B3 64G×4；TP4；BF16；`max_model_len=8192`、`max_num_seqs=32`、gmu 0.80、seed 0、关 prefix cache | 与上行相同的五组 ISL/OSL/batch；按模型词表生成确定性随机 token | `s3fp÷s2`；greedy 定长；每负载 warmup 1 次后 3 次取中位数 | **1.11 / 1.11 / 1.23 / 1.26 / 1.17**。[Stage6](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md#二头条s3fp--s2各模型最好形态-vs-生产默认>) |

参数与输入的代码依据：[性能脚本模型表](<../../../inductor-meta-worktree/docs/vllm/env/scripts/bench_inductor_perf.py#L33>)、[五负载](<../../../inductor-meta-worktree/docs/vllm/env/scripts/bench_inductor_perf.py#L86>)、[引擎参数](<../../../inductor-meta-worktree/docs/vllm/env/scripts/bench_inductor_perf.py#L127>)。

## 6. 冷启动、暖启动和缓存复用

| 测试项 | 测试环境与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|
| Qwen3-0.6B 暖启动 | Stage5 历史环境；同一 Inductor 配置连续启动，保留首启缓存 | 同配置生成；干净暖启记录首 generate 32 tokens | 比较两启 init；检查 `Directly load`、compilation_time、缓存文件/mtime 和输出 | 源树干净暖启 init 61.3s、compilation_time=0、图加载约 1.68s；Stage6 社区口径冷 518.7s、暖 47.4s。[Stage5 P0-1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>)、[Stage6 启动](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |
| Qwen3-8B BF16 暖启动 | Stage5 历史环境；同配置二连启 | 具体 prompt 在汇总 RESULTS 未记录 | 同上行，检查 Directly load、编译计数和缓存不增 | 历史暖启 71.3s，Directly load 约 1.878s，compilation_time=0。[Stage5 M1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| Qwen3-8B-W8A8 暖启动 | Stage5 历史环境；同配置二连启，NZ=0 | 具体 prompt 在汇总 RESULTS 未记录 | 同上行 | 冷启 515.1s、暖启 64.1s，约 8.0×，缓存零新增。[Stage5 M1](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| 缓存 key 隔离 | Stage2 历史环境；`TORCHINDUCTOR_NPU_BACKEND=default/triton_experimental`；`cg=NONE/PIECEWISE` | 同一 Qwen3-0.6B 配置与输入 | 改变后端和图模式，比较 cache key、目录和是否重新冷编；二次启动检查输出一致 | 后端和图配置均有缓存隔离证据；这只覆盖普通 FxGraphCache，不等于 AOT/损坏缓存回退通过。[Stage2 T3](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage2/04_测试与验收.md>) |

## 7. A5 构建与冒烟

| 测试项 | 测试环境 | 模型与运行参数 | 输入 | 测试方法 | 结果与证据 |
|---|---|---|---|---|---|
| A5 构建与 eager/Inductor 冒烟 | Ascend950DT；CANN 9.2.0；Python 3.12.16；torch 2.13.0+cpu；torch_npu 2.13.0；triton-ascend 3.6.0.post2 | Qwen3-0.6B；eager 与 `backend=inductor` 两路；其余引擎参数在部署汇总未完整记录 | 英文首都、中文长城两条请求；后续另记录单请求 8 tokens | 先构建 custom aclnn/vllm_ascend_C，再分别生成；检查 TE 缓存文件和 `.npubin` | eager 非空；Inductor 冷编 406.70s 并打印 PASS；10 个 TE marker 文件、98 个 `.npubin`。只证明基本链路，不证明多卡、量化、长序列、完整精度或性能。[A5 部署记录](../a5-25-cann920-env-deploy.md) |

## 8. 单元测试和回归门禁

这些数字对应不同时间和测试选择集，不能当成同一集合的简单递增。

| 阶段 | 测试环境与选择集 | 方法 | 历史结果与证据 |
|---|---|---|---|
| Stage1 | 910B2 主机；compilation 配置 UT，含既有零回归 | 测轨开关、后端/pass 配置、legacy 兼容和基础编译配置 | **48 passed**，含既有 28 个零回归。[Stage1 里程碑](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage1/05_实施步骤与里程碑.md>) |
| Stage2 | `tests/ut/compilation/ + test_ascend_config.py` | 增加枚举、冲突、PIECEWISE 默认、缓存、RNG 和 TP 配置 | **157 passed**。[Stage2 T2 结果](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage2/_notes/t2_evidence/RESULTS.md>) |
| Stage3 | 同上 | 增加 FULL 族、融合 Pass/UUID/match_table、legacy cache 修复 | **172 passed**。[Stage3 里程碑](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage3/05_实施步骤与里程碑.md>) |
| Stage4 初结项 | compilation、ascend_config、lora | 增加模型配置、量化 pattern、Meta、配置治理和可观测性 | **272 passed**。[Stage4 里程碑](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage4/05_实施步骤与里程碑.md>) |
| Stage4 batch2 | 扩大后的四套件口径 | 增加 RoPE、GDN、配置单例、counter 等回归 | **362 passed + 1 skipped**。[CHECKPOINT](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/CHECKPOINT.md>) |
| 配置重构及回退后 | compilation、ascend_config、lora、platform | 上游 backend 正门、删除枚举和 legacy 修复回退 | 最终记录 **353 passed + 1 skipped**。[配置重构审计](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/vllm config refactor/06_对抗性复审_只动inductor原则.md>) |
| Stage5 | `-k "compilation or ascend_config or lora"`，另有 worker 专项 | custom_ops、warmup、F4（当时）、默认策略 | **292 passed**，另有 worker **4 passed**。[Stage5 结果](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage5/_notes/t0_probe/RESULTS.md>) |
| F4 回退后 | compilation 与 ascend_config 分开统计 | 删除 F4 及其 UT 后复验 | compilation **113 passed**；ascend_config **92 passed**。[CHECKPOINT F4 回退](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/CHECKPOINT.md>) |

## 9. 此前视觉/小模型批测

这些不是 vLLM InductorAdaptor 大模型主链的测试。每行环境均为 **910B2 单卡**，方法均为**编译版与 eager 的精度和性能对比**；原汇总没有记录具体输入张量形状，因此输入列统一标为“原文未记录”。

| 模型 | 输入 | 测试方法与结果 | 证据 |
|---|---|---|---|
| adv_inception_v3 | 原文未记录 | 编译/eager 性能比 1.003；精度通过 | [本地验证情况](../03-本地验证情况.md) |
| convnextv2_nano | 原文未记录 | 0.967；精度通过 | [同上](../03-本地验证情况.md) |
| deit_base_distilled | 原文未记录 | 1.007；精度通过 | [同上](../03-本地验证情况.md) |
| mnasnet1_0 | 原文未记录 | 1.086；曾精度失败，归因为 BatchNorm 分解链 ulp 差被 15 层放大，按记录已修复 | [同上](../03-本地验证情况.md) |
| maml_omniglot | 原文未记录 | 1.096；精度通过 | [同上](../03-本地验证情况.md) |
| maml | 原文未记录 | 0.634；精度失败，偏差 133 倍 | [同上](../03-本地验证情况.md) |
| mobilenet_v2_quantized_qat（训练态） | 原文未记录 | 0.887；精度通过且逐位一致 | [同上](../03-本地验证情况.md) |
| mobilenetv3_large_100 | 原文未记录 | 0.803；精度通过 | [同上](../03-本地验证情况.md) |
| mobilevit_s | 原文未记录 | 0.210；精度失败，偏差 287 倍，融合核数值问题 | [同上](../03-本地验证情况.md) |
| beit_base | 原文未记录 | 0.094；精度通过，单个 bool mask 融合核占约 92% 耗时 | [同上](../03-本地验证情况.md) |
| YituTechConvBert | 环境中无模型 | 未测试 | [同上](../03-本地验证情况.md) |

同日还针对 grid 超过 65535 的 AutoBlockify 编译崩溃做了修复验证，并累计修复 13 项单元测试问题。

## 10. 尚未形成有效测试结论的场景

| 场景 | 当前准确口径与来源 |
|---|---|
| EPLB × Inductor | 没有组合验证，不能由 MoE 通过外推。[覆盖缺口](../feature/Inductor适配功能性能与覆盖缺口.md) |
| GLM 系列 | 最小权重和 TP8/可用卡数不满足，Stage6 未开模型。[Stage6 范围](<../../../inductor-meta-worktree/docs/vllm/vllm_ascend_inductor_design/stage design/stage6/RESULTS.md>) |
| medusa、PARD/draft_parallel、dspark 等推测解码 | 缺权重或本地配方，未测；个别 MTP 模型通过不代表全部 MTP 配方通过。[Spec 测试说明](<../../../inductor-meta-worktree/vllm-ascend/tests/e2e/pull_request/one_card/inductor/test_inductor_track_spec.py>) |
| 多模态 encoder、非默认 dynamic shapes | 没有对应 Inductor 组合测试。[覆盖缺口](../feature/Inductor适配功能性能与覆盖缺口.md) |
| A5 多卡、量化、长序列、完整精度和性能 | 当前只有第 7 节的小模型构建与冒烟，不能外推。[A5 部署记录](../a5-25-cann920-env-deploy.md) |

## 11. 总结

已有工作不是单一“能跑通”的 smoke，而是由配置 UT、真实模型 E2E、图计数、数值对拍、编译产物、融合命中、缓存二连启、性能矩阵、最小复现和 A5 构建冒烟组成。

但每行结论都只对所列环境、参数、输入和判据成立。当前最需要补的是：

1. 修正第 3 节标出的 backend/NZ/冷缓存配置后重跑；
2. 将既有 910B 测试转化为可重复的真实 CI，而不是只保留历史日志；
3. 在 A5 上按同样逐行台账补多卡、量化、长序列、精度和性能；
4. 对 xfail、skip、底层 eager 阻塞和未测试场景保持独立状态，不能汇总成“全部通过”。

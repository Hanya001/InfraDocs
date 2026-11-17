# DeepSeek-V4-Flash-FP8：TP4 下稀疏注意力算子 Head 数限制

## 1. 当前状态

**状态：暂不解决。**

当前已完成直接根因定位和社区资料核对，没有修改产品源码，也没有形成修复后的端到端通过记录。该问题不是 DeepSeek-V4 模型未受支持，也不能笼统归因于 vllm-ascend 0.27；已确认的是 **A5 FP8 KV sparse-attention 路径与 TP4 切分后的每 rank head 数不兼容**。

## 2. 环境与模型配置

- 硬件：A5 / Ascend950DT。
- 模型：`/mnt/weight/DeepSeek-V4-Flash-FP8`。
- 权重：46 个 safetensors 分片，索引引用完整，约 148.66 GiB。
- 软件：vLLM 0.27.1、vllm-ascend 0.27.1rc0、torch/torch-npu 2.13、triton-ascend 3.6.0.post2、CANN 9.2.0。
- 运行配置：TP4、卡 0–3、`enable_expert_parallel=True`、`tokenizer_mode=deepseek_v4`、`kv_cache_dtype=auto`、`max_model_len=8192`。

模型关键配置：

```text
num_attention_heads=64
num_key_value_heads=1
head_dim=512
index_topk=512
sliding_window=128
```

这些模型维度本身并不是异常值。对应 FP8 KV 算子文档也明确列出 Ascend950PR/950DT 支持，但 `num_heads_q` 目前只支持 64/128。

## 3. 实测现象

| 路径 | 失败阶段 | 直接证据 |
|---|---|---|
| 默认编译路径 | 首次 profiling | `KvQuantSparseAttnSharedkv` tiling 失败，日志明确给出 `got 16` |
| Inductor | 首次 profiling | 与默认路径相同，不是 Inductor 独有问题 |
| eager，`enforce_eager=True` | 初始化成功 156.27 秒，第一次生成失败 | 同一 head 数错误，同时出现 metadata AI CPU kernel 失败和后续同步错误 |

三条路径均未生成有效文本，也没有正式性能数据，因此不能报告精度或加速比。

原始日志：

- [默认路径](../../../server-scripts/bench213/20261114_A5机器模测/20261114_8models_01/DeepSeek-V4-Flash-FP8/default.runlog)；
- [Inductor 路径](../../../server-scripts/bench213/20261114_A5机器模测/20261114_8models_01/DeepSeek-V4-Flash-FP8/inductor.runlog)；
- [eager 路径](../../../server-scripts/bench213/20261117_A5机器模测_DeepSeekV4_Eager/20261117_deepseek_v4_eager_01/DeepSeek-V4-Flash-FP8/default.runlog)。

日志中的核心错误为：

```text
n1Size_ only support 64 and 128 now, but got 16.
Failed to execute operator KvQuantSparseAttnSharedkv. Reason: Failed to execute tiling function.
```

## 4. 调用链与直接根因

vllm-ascend 的 DSA 路径按 TP 计算每 rank 的 head 数：

```python
n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
```

对应 [dsa_v1.py:685](../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L685)，随后把该值作为 `num_heads_q` 传给 metadata 算子。

本次 TP4 因而得到：

```text
64 / 4 = 16
```

A5 上 `kv_cache_dtype=auto` 会选择 FP8 KV 计划，调用 `npu_kv_quant_sparse_attn_sharedkv` 和对应 metadata 接口，见 [dsa_attn_kv_plan.py:170](../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/attention/dsa_attn_kv_plan.py#L170)。

底层 tiling 检查在 [kv_quant_sparse_attn_sharedkv_check_single_para.cpp:259](../../../inductor-meta-worktree/vllm-ascend/csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_single_para.cpp#L259) 明确拒绝非 64/128 的输入：

```cpp
OP_CHECK_IF(n1Size_ != 64 && n1Size_ != 128,
    OP_LOGE(opName_, "n1Size_ only support 64 and 128 now, but got %u.", n1Size_),
    return ge::GRAPH_FAILED);
```

因此直接根因是：**TP4 将 64 个 Q head 切成每 rank 16 个，而当前 FP8 KV sparse-attention 算子只接受 64/128。** eager 只改变编译和图捕获方式，不会改变这个算子约束。

## 5. 为什么不是简单的“版本不支持模型”

- 官方支持矩阵明确列出 DeepSeek V4-Flash 对 Ascend 950 Products 的支持，见 [supported_models.md:22](../../../inductor-meta-worktree/vllm-ascend/docs/source/user_guide/support_matrix/supported_models.md#L22)。
- 当前源码具有完整模型、DSA 和算子调用实现，不是模型注册缺失。
- 故障发生在特定 FP8 KV＋TP4 组合的底层算子约束上，不能外推为 0.27 或所有 DeepSeek-V4 路径都不可用。
- 当前工作区还有本地修改，且没有进行同配置跨版本对照，不能把问题定性为所有官方 0.27 环境的普遍回归。

更准确的定性是：**当前版本在该模型、硬件、KV 精度和 TP 组合上存在未覆盖或未成文的关键限制。**

## 6. 社区相近讨论及其边界

社区 issue [vllm-ascend #21395](https://github.com/vllm-project/vllm-ascend/issues/21395) 同样涉及 A5、64-head DeepSeek-V4.1、TP>1 和 metadata kernel 的固定 64-head 要求。

该 issue 最初认为应把硬编码的 64 改为 TP 后的 `n_local_heads`，但作者在真实 A5 TP4 上验证后撤回了这一判断：[最终评论](https://github.com/vllm-project/vllm-ascend/issues/21395#issuecomment-6759613374) 明确指出，改成 16 后每次失败，而原来的 64 可以成功。因此 64 是该 metadata kernel 的固定要求，不是单纯的 Python 硬编码错误。

该讨论与当前问题相互印证了固定 head 数约束，但存在重要边界：

- 社区路径是 DeepSeek-V4.1 的 `mixed_quant_sparse_flash_mla_metadata`；
- 当前路径是 DeepSeek-V4 的 `KvQuantSparseAttnSharedkv`；
- 不能据此直接断定把当前调用改成 64 就能正确运行，更不能只删除参数校验。

[v0.27.1rc1 release checklist](https://github.com/vllm-project/vllm-ascend/issues/16288) 还将 DeepSeek-V4-Flash on 950DT 纳入发布范围，并列出 SparseFlashMla BF16 KV 支持。这说明存在经过验证的模型路径，但不代表当前 FP8 KV＋TP4 组合已经覆盖。

## 7. 可行方向与暂不解决的原因

### 7.1 为什么仅调整 TP 不合适

- TP1 可保持 64 heads，但约 149 GiB 权重无法放入单张 96 GiB 卡。
- TP2 会得到 32 heads，TP8 会得到 8 heads，仍不符合当前算子的 64/128 限制。
- 因此在这台 8×96 GiB A5 上，不能靠简单增减 TP 找到当前 FP8 KV 路径的可用配置。

### 7.2 候选替代路径

A5 已有 `cache_dtype="bfloat16"` 时选择 SparseFlashMla 的实现，见 `dsa_attn_kv_plan.py` 和 `sparse_flash_mla.py`。该路径可能避开当前 FP8 KV 算子，但会改变 KV 显存、数值和性能口径，必须另行执行精度和性能验证。目前尚未进行该实测，不能写成已验证规避方案。

### 7.3 真正修复所需工作

若继续使用当前 FP8 KV 算子，需要：

1. 确认 metadata 的 head 参数语义究竟是全局 head 数还是 rank-local head 数；
2. 确认底层 kernel 是否能支持 16/32 heads，而非仅修改上层校验；
3. 如需调整并行映射或任务划分，补充 TP2/TP4/TP8 的输出正确性和性能回归；
4. 将默认、Inductor、eager 三条路径都纳入验证。

当前没有足够证据支持一个安全、通用且可上游化的修复，因此状态保留为 **暂不解决**。

## 8. 与追踪表其他条目的关系

- 与 E-01 的 DSV4 权重名实不符和 OOM 不同；本条权重完整且已成功加载，失败发生在 sparse-attention 执行阶段。
- 与 O-05 的生成收尾段超时不同；本条在首次 profiling 或首次生成时即失败。
- 与 Inductor 编译器本体无关；默认路径和 eager 均可复现相同约束错误。

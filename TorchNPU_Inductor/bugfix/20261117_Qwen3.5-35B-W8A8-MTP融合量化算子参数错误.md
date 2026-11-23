# Qwen3.5-35B-A3B-W8A8-MTP：融合量化算子参数错误导致 eager 与编译路径均无法启动

## 1. 当前状态

**状态：eager跑不通 暂不解决。**

该问题已在 A5 / Ascend950DT、TP2 环境中稳定复现。默认编译路径、Inductor 路径和 `enforce_eager=True` 路径都在初始显存 profiling 阶段失败，尚未进入正常请求推理。当前只完成根因定位和接口核对，没有修改产品源码，也没有形成修复后的端到端通过记录。

本问题不是 MTP 独有问题，也不是 Inductor 编译器独有问题。MTP 关闭时同一 W8A8 权重同样失败，开启 MTP 只是本次模型验证场景。

## 2. 环境与现象

- 模型：`/mnt/weight/Qwen3.5-35B-A3B-w8a8-mtp`。
- 硬件：A5 / Ascend950DT，使用 2 卡 TP。
- 软件：vLLM 0.27.1、vllm-ascend 0.27.1rc0、torch/torch-npu 2.13、triton-ascend 3.6.0.post2、CANN 9.2.0。
- 失败阶段：EngineCore 调用 `determine_available_memory`，worker 执行初始 profiling。
- 直接异常：

```text
RuntimeError: npu::npu_grouped_matmul_swiglu_quant_v2()
Expected a value of type 'List[Tensor]' for argument 'weight'
but instead found type 'Parameter'.
```

后续出现的 `Worker proc ... died unexpectedly` 或 benchmark 退出码 143，都是该异常触发的进程清理结果，不是独立根因。

## 3. 调用链与根因

实际调用链为：

```text
AscendW8A8DynamicFusedMoEMethod.apply
  → moe_comm_method.fused_experts
  → unified_apply_mlp / quant_apply_mlp
  → DeviceOperator.npu_grouped_matmul_swiglu_quant
  → torch_npu.npu_grouped_matmul_swiglu_quant_v2
```

对应源码位置：

- [w8a8_dynamic.py:230](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py#L230>)；
- [moe_mlp.py:534](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py#L534>)；
- [device_op.py:1015](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/device/device_op.py#L1015>)。

`quant_apply_mlp` 在调用设备适配层前，通过 `_require_single_tensor_for_swiglu_quant` 将单元素权重列表解包为单个 tensor。A5 的 INT8 分支随后直接传入：

```python
weight=weight,
weight_scale=weight_scale,
```

但当前 torch-npu 注册的算子签名是：

```text
npu::npu_grouped_matmul_swiglu_quant_v2(
    Tensor x,
    Tensor[] weight,
    Tensor[] weight_scale,
    Tensor x_scale,
    Tensor group_list,
    ...
) -> (Tensor, Tensor)
```

因此首个确定根因是：**W8A8 INT8 分支没有把单个权重和权重 scale 重新封装成算子要求的单元素列表。** 同文件 FP8 分支在 [device_op.py:1004](<../../../inductor-meta-worktree/vllm-ascend/vllm_ascend/device/device_op.py#L1004>) 已使用 `weight=[weight]` 和 `weight_scale=[weight_scale]`，可作为接口用法对照。

## 4. 同一分支还包含无效关键字

INT8 分支还向 `npu_grouped_matmul_swiglu_quant_v2` 传入：

```python
swiglu_limit=swiglu_limit,
use_mxfp_quant=False,
```

这两个参数均不在当前 torch-npu 算子签名中。分别进行直接调用验证，得到：

```text
RuntimeError: Unknown keyword argument 'swiglu_limit'
RuntimeError: Unknown keyword argument 'use_mxfp_quant'
```

所以不能只修复第一个 `weight` 类型错误就宣称问题闭环。后续若恢复处理，应同时核对并修正整个调用的参数列表；`bias` 在当前签名中是合法参数。

候选接口修正方向如下，但**尚未实施或验证**：

```diff
 return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
     x=x,
-    weight=weight,
+    weight=[weight],
     group_list=group_list,
-    weight_scale=weight_scale,
+    weight_scale=[weight_scale],
     x_scale=x_scale,
     bias=bias,
-    swiglu_limit=swiglu_limit,
-    use_mxfp_quant=False,
 )
```

接口修正后仍需继续检查实际 shape、NZ 权重格式、量化参数和数值输出，不能预先认定该改动足以使模型完整跑通。

## 5. 复现记录

| 日期 | 路径 | MTP 配置 | 结果 |
|---|---|---|---|
| 2026-11-16 | 默认路径 | 关闭，`speculative_config=None` | profiling 中报相同的 `weight` 类型错误 |
| 2026-11-16 | Inductor | 关闭 | 同上 |
| 2026-11-16 | 默认路径 | 开启，`num_speculative_tokens=1` | 同上 |
| 2026-11-16 | Inductor | 开启，`num_speculative_tokens=1` | 同上 |
| 2026-11-17 | 默认路径，当前服务器源码复跑 | 开启，`num_speculative_tokens=1` | 两个 TP worker 均复现相同异常 |
| 2026-11-17 | eager，`enforce_eager=True` | 开启，`num_speculative_tokens=1` | `CompilationMode.NONE`、`CUDAGraphMode.NONE` 下仍复现相同异常 |

2026-11-16 的汇总见：

- [MTP 关闭复测](../../../server-scripts/bench213/20261116_A5机器模测_Qwen35修复/20261116_qwen35_fix_01/RESULTS.md)；
- [MTP 开启复测](../../../server-scripts/bench213/20261116_A5机器模测_Qwen35修复/20261116_qwen35_w8a8_mtp_on_01/RESULTS.md)。

2026-11-17 的服务器诊断日志位于：

- `/home/z00896543/scripts/bench213/a5/20261117_qwen35_w8a8_mtp_diagnosis_01/Qwen3.5-35B-A3B-w8a8-mtp/default.runlog`；
- `/home/z00896543/scripts/bench213/a5/20261117_qwen35_w8a8_mtp_diagnosis_01/eager.runlog`。

人工 eager 复现可在已加载 `env.sh` 的容器终端前台执行，不重定向或隐藏日志：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

vllm serve /mnt/weight/Qwen3.5-35B-A3B-w8a8-mtp \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

如果 CLI 在模型加载前因缺少 `model_hosting_container_standards` 而退出，那是追踪表 F-13 已覆盖的独立依赖问题，不属于本条算子故障。

## 6. 与已有记录的关系及后续条件

- 与 R-01 的 Qwen3.5 GDN LayerNorm/`eps` 修改无关；当前失败发生在 MoE W8A8 融合算子调用处。
- 与 O-02 的 near-tie 数值分叉不同；本条在任何有效输出产生前即失败。
- 与 O-10 的 `aclnnSwiGluQuantV2` 动态模式 AIV/MTE 越界不同；本条首先是 Python 到 torch 算子 schema 的参数类型和关键字不匹配。
- 历史上 910B3 的同名模型性能记录不能证明当前 A5、torch-npu 和 vllm-ascend 组合可用，也不能替代本次失败结论。

当前决定为 **eager跑不通 暂不解决**。后续恢复处理时，至少应先验证 eager 启动和短生成，再分别复测 MTP 开/关、默认路径和 Inductor 路径，并检查生成内容及失败进程清理；在这些验证完成前不得标记为已修复。

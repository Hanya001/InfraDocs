# VF Fusion Mode 接入 vLLM Ascend：有效配置、真实编译证据与价值评估

## 1. 阶段状态

- 日期：2026-11-22。
- 服务器：A5-25，Ascend950DT，CANN 9.2.0。
- 容器：`inductor_feature_z00896543_20261122_140509`。
- 分支：`hanya/feat/inductor-low-cost-fusions`。
- 功能提交：`7d8d0740f19b113b68a3cbc48743cb7ccf6f149b`，对应 `Patch/VFFusion/0004-expose-vf-fusion-mode-on-npu.patch`。
- 相关格式整理：`c8af34cae813691eecba0681cd96fab7b0686240`，对应 `Patch/MMRS/0008-format-mmrs-and-vf-changes.patch`。
- 实现结论：接入的是当前编译器实际消费的 `vf_fusion_mode`，不是已被忽略的 `enable_vf_fusion`。独立 TE 冒烟已证明配置进入 4 次真实编译，输出与 eager 一致。大模型性能数据见后文。

## 2. 为什么不能只透传 `enable_vf_fusion`

torch_npu 配置模块同时提供：

- `enable_vf_fusion`；
- `vf_fusion_mode`；
- `vf_merge_level`。

最初判断只需把总开关传给 Triton 即可。独立冒烟中，`triton.compile` 确实收到了 `enable_vf_fusion=True`，但当前 triton-ascend 给出明确警告：该选项已废弃并被忽略，不再转发到 NPU 编译器。因此“配置到达 Python 调用边界”不能证明 VF Fusion 生效。

进一步检查当前编译器代码后确认：

- `enable_vf_fusion` 是 `init=False` 的兼容字段，会在选项归一化时删除；
- `vf_fusion_mode` 仍是公开选项；
- 非空 mode 会形成 `--vf-fusion-mode=<value>` 并传给 bishengir；
- Triton 编译路径会运行 VF Fusion pass，mode 决定策略。

因此最终只新增：

```python
additional_config={
    "ascend_compilation_config": {
        "vf_fusion_mode": "max-parallel",
    }
}
```

允许值为 `ub-aware-op` 和 `max-parallel`。默认 `None`，保持编译器原有选择，不改变默认行为。

## 3. 实现

修改包括：

- `vllm_ascend/ascend_config.py`：用 `Literal` 限制 mode 的合法取值；
- `AscendPostGradPassManager._inject_ascend_fusion_passes()`：在代码生成前，将配置写入 torch_npu 的 triton_experimental 配置对象；
- 默认配置再次写回 `None`，避免同一进程内后续配置继承上一次显式值；
- 旧 torch_npu 不支持该配置时，只在用户显式设置 mode 后告警；
- `additional_config.md`：说明合法值、默认行为以及不暴露旧总开关的原因；
- 单元测试：验证 `max-parallel` 能传递到 torch_npu，并能在下一次默认配置时复位。

没有修改 torch_npu、Triton kernel 或 bishengir，也没有增加环境变量。

## 4. 单元测试

最终相关 vLLM Ascend 与 MMRS 测试合并执行，结果：`250 passed`，原始日志为 `server-results/inductor-feature/20261122_140509/final_unit.log`。VF 相关测试验证：

- mode 配置能写入 torch_npu；
- 默认配置能恢复 `None`；
- 既有 Ascend pass 注册顺序和条件不受影响。

torch_npu 的 `test_package.py` 全部执行，结果：`12 passed`，覆盖后端配置序列化、编译选项和 kernel 打包行为。

## 5. 独立 TE 真实编译证据

测试函数返回两个独立 pointwise 输出：

```python
def pointwise_pair(x, y):
    return x.relu(), y.sigmoid()
```

测试在 `te_config.patch(vf_fusion_mode="max-parallel")` 下编译，并包装 `triton.compile` 记录实际选项。禁用缓存后，日志为：

```text
VF_COMPILE_OPTION vf_fusion_mode=max-parallel
VF_COMPILE_OPTION vf_fusion_mode=max-parallel
VF_COMPILE_OPTION vf_fusion_mode=max-parallel
VF_COMPILE_OPTION vf_fusion_mode=max-parallel
VF_SMOKE_PASS compile_calls=4
```

两个输出均通过 `torch.testing.assert_close` 与 eager 比较。该结果证明：

- 配置不是只存在于 vLLM 配置对象中；
- mode 进入了 4 次真实 Triton 编译；
- 编译后的数值结果正确。

旧 `enable_vf_fusion` 的警告仍可能由 torch_npu 的兼容字段触发，但本次有效证据是 `vf_fusion_mode`，不再依赖旧字段。

## 6. 大模型 A/B 性能与正确性

### 6.1 Qwen3-32B 启动异常，不作为性能数据

最初使用 Qwen3-32B 单卡验证。默认组完成 Inductor 编译后，在初始 profiling/warmup 阶段停留近 50 分钟，未进入生成，也未产生 `result.json`；同期 NPU 仍有活动，因此不能简单归类为 Python 死锁。该次运行已终止并释放 NPU，原始日志保存为 `vf_fusion_qwen32b/off/run.stalled.log`。它不用于判断 VF 的性能或正确性。

### 6.2 Qwen3.8-27B 单卡 A/B

改用目标模型级别内的 Qwen3.8-27B，两组均使用单卡、BF16、16 条提示、每条最多 128 token、一次 warmup 后测量三轮，prefix caching 开启，编译缓存按组隔离：

- 关闭组：`vf_fusion_mode=None`；
- 开启组：`vf_fusion_mode="max-parallel"`；
- 其余模型、提示、warmup 和生成参数一致。

结果：

| 指标 | 默认 `None` | `max-parallel` | 差异 |
|---|---:|---:|---:|
| 启动时间 | 342.450 s | 353.476 s | 开启增加 11.026 s（3.22%） |
| 第 1 轮生成 | 3.226006 s | 3.212519 s | — |
| 第 2 轮生成 | 3.202833 s | 3.194713 s | — |
| 第 3 轮生成 | 3.208861 s | 3.212821 s | — |
| 三轮生成中位数 | 3.208861 s | 3.212519 s | 开启增加 0.003658 s（0.11%） |
| 首轮输出 SHA256 | `302d5d88534b0c688d215ed1694b64e932d4388957bbcbfa46770c0f96156e77` | 相同 | 文本一致 |

两组样例均为围绕太阳能光伏发电的正常中文回答，首轮全部 16 条输出的拼接 hash 一致。两组各自的三轮输出都存在一些跨轮差异，因此不把 `temperature=0` 等同于端到端必然确定性；本次 A/B 的首轮结果一致，没有发现 VF 引入的额外首轮输出变化。

默认组先于 MMRS 默认门控提交运行，开启组后于该提交运行。两组唯一的源码差异是新增 `fuse_mm_reduce_scatter` 配置；它在开启组结果中显式为 `False`，且本次为单卡非 SP，不改变 VF 的计算路径。

**收益结论：`max-parallel` 在本模型和负载下没有明确稳态收益。** 中位数仅慢 0.11%，远小于三轮波动，应视为未观察到可区分的变化；单次冷启动慢 3.22%，不能包装成优化。默认值继续保留为 `None`，由编译器选择策略。`max-parallel` 作为显式实验选项保留，后续只有在特定 shape 的独立 kernel 或大模型 A/B 显示稳定收益时，才建议针对该场景启用。

## 7. 使用、回退与原始结果

使用：

```python
additional_config={
    "ascend_compilation_config": {
        "vf_fusion_mode": "max-parallel",
    }
}
```

恢复默认：删除该配置，或设为 `None`。

Patch：

```text
doc/InfraDocs/TorchNPU_Inductor/feature/Patch/VFFusion/0004-expose-vf-fusion-mode-on-npu.patch
```

独立 TE 冒烟结果位于服务器 `results/vf_smoke/max_parallel.log`，本地副本为 `server-results/inductor-feature/20261122_140509/vf_max_parallel.log`。大模型结果归档目录为 `server-results/inductor-feature/20261122_140509/vf_fusion_qwen27b/`；Qwen3-32B 的异常启动日志单独保留在 `server-results/inductor-feature/20261122_140509/vf_fusion_qwen32b/`。

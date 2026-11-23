# Dynamic RMSNorm+Quant view 变体验证：覆盖、收益与默认策略

## 1. 阶段结论

- 日期：2026-11-22。
- 服务器：A5-25，Ascend950DT，CANN 9.2.0。
- 容器：`inductor_feature_z00896543_20261122_140509`。
- 工作区：`/home/z00896543/validation/inductor_feature_20261122_140509/inductor-meta-worktree`。
- 分支：`hanya/feat/inductor-low-cost-fusions`。
- 阶段提交：`7fc0fed88421587b65b1275920b0a2273c045f95`。
- 对应 patch：`Patch/DynamicNormQuant/0002-cover-dynamic-norm-quant-flag-forwarding.patch`。
- 结论：配置和 pattern 已存在，本阶段补充了配置传递单元测试；在 Qwen3.6-27B-W8A8 和 Qwen3-0.6B-W8A8 上均未发现 view 专用 pattern 的独立命中。现有 direct pattern 已覆盖实际图形态，因此本变体没有展示出增量价值，继续保持默认关闭。

这里不能把开启前后的启动或生成耗时差异归因于该变体，因为专用 pattern 没有命中。

## 2. 背景与已有实现

W8A8 动态量化模型中，Norm 输出到动态量化之间可能出现等形的 `view` 或 `reshape`：

```text
AddRMSNorm(bias=None) -> view/reshape -> npu_dynamic_quant
```

`AddRMSNormDynamicQuantPatternWithNoneBiasView` 的目标是把上述链替换为 `npu_add_rms_norm_dynamic_quant`，再恢复量化输出形状。

当前代码在本阶段开始前已经具备：

- `ascend_compilation_config.fuse_norm_quant_dynamic`，默认 `False`；
- Inductor 路径向 `AddRMSNormQuantFusionPass` 传递该配置；
- view/reshape 两种手写搜索 pattern；
- 默认 direct `AddRMSNormDynamicQuantPatternNoBias` pattern。

因此本阶段不重写融合实现，只验证配置传递、真实图覆盖和收益。

本阶段新增的测试位于 `tests/ut/compilation/test_inductor_track_fusion_chain.py`：配置为真时，Inductor pass manager 创建出的 fusion pass 必须带有 `dyn_fusion=True`，防止配置入口存在但传递中断。

## 3. 单元测试

执行：

```bash
cd /home/z00896543/validation/inductor_feature_20261122_140509/inductor-meta-worktree
source env.sh
cd vllm-ascend
$PY -m pytest -q \
  tests/ut/compilation/test_norm_quant_fusion_w2_patterns.py \
  tests/ut/compilation/test_inductor_track_fusion_chain.py
```

结果：`17 passed`。测试覆盖 direct 融合、view/reshape pattern、错误形状拒绝、默认 direct 路径以及本阶段新增的配置传递检查。

## 4. 真实模型覆盖证据

### 4.1 观测方法

两组使用独立 `VLLM_CACHE_ROOT`，模型、提示、生成参数和其他编译配置一致。日志记录 `rmsnorm_quant_fusion_pass` 的总命中数。

为区分 direct pattern 与 view 专用 pattern，验证期间在 view replacement 内临时加入 `DYNAMIC_NORM_QUANT_VIEW_FUSION_HIT` 日志探针。探针不改变替换逻辑，且已从服务器和本地源码中移除，不属于阶段提交。

### 4.2 Qwen3.6-27B-W8A8

模型：`/mnt/weight/Qwen3.6-27B-W8A8`，单卡 BF16，16 条提示，每条最多生成 128 token。

| 项目 | 关闭组 | 开启组 |
|---|---:|---:|
| `rmsnorm_quant_fusion_pass` 总命中 | 6 | 6 |
| view 专用 pattern 增量命中 | 无增量 | 无增量 |
| 首轮输出 SHA256 | `0ea2213b07cfcb498494e950277d55341819365ad1132277a7447ed702b402bb` | 相同 |
| 冷启动时间 | 393.208 s | 327.607 s |

两组总命中完全相同，开启组没有新增融合。27B 首轮编译完成后才补入专用探针，因此这里只把“总命中无增量”作为证据，不虚构 27B 的专用探针计数；0.6B 运行则完整覆盖探针，直接确认 view 命中为零。冷启动差异只记录为单次运行波动，不作为特性收益。

### 4.3 Qwen3-0.6B-W8A8

模型：`/mnt/weight/Qwen3-0.6B-W8A8`，单卡，4 条提示，每条最多生成 32 token，一次 warmup 后执行一轮生成。

- 总命中：3；
- view 专用 pattern 命中：0；
- 输出非空，输出 SHA256：`e882c10b1660ba473b5c7598472f8a9d42dec9ca0d7e5f13556ab2e760ac53d1`；
- 启动时间：367.210 s；
- warmup 后生成时间：0.182 s。

小模型同样只有 direct 命中，没有 view 专用命中。

## 5. 为什么没有独立收益

上游 `PostGradPassManager.configure()` 先注册 `NoOpEliminationPass`，Ascend 融合 pass 后运行。`NoOpEliminationPass` 会删除输入、输出形状相同的 reshape 和 slice。

因此，实际图中的等形 view 在 Ascend Norm+Quant pass 之前已经被删除，后续 direct pattern 即可完成融合：

```text
AddRMSNorm(bias=None) -> npu_dynamic_quant
```

view 专用 pattern 所要处理的情况在当前 pass 顺序下不再出现。两个不同规模的模型都符合这一解释：总命中只来自 direct pattern，view 探针始终为零。

这也说明“存在手写 view pattern”并不等于它在当前编译链中有独立价值。旧 npugraph_ex 路径的 pass 顺序不同，本文不把该结论外推为“所有后端都应删除此 pattern”；这里只评价 Inductor 路径。

## 6. 稳定生成耗时

为减少首轮动态编译和前缀缓存预热影响，后续脚本增加一次独立 warmup，再测量三轮生成。

| 指标 | 关闭组 | 开启组 |
|---|---:|---:|
| 启动时间（本批运行） | 321.263 s | 320.850 s |
| 三轮生成时间 | 2.379634 / 2.356009 / 2.360590 s | 2.371903 / 2.377164 / 2.353524 s |
| 生成中位数 | 2.360590 s | 2.371903 s |

开启组生成中位数约慢 0.48%，启动时间基本持平。由于 view 专用 pattern 没有命中，这些差异只反映运行波动，不能归因于特性开启或关闭；本批数据没有展示性能收益。两组首轮输出 SHA256 相同。

## 7. 默认策略与后续建议

- 保持 `fuse_norm_quant_dynamic=False`，不将本变体作为性能优化默认开启。
- 不删除已有 direct 融合；两个模型上的 direct 融合都是真实生效的。
- 暂不删除 view pattern，以免影响使用不同 pass 顺序的旧编译路径。
- 后续若模型图包含不能被 NoOpElimination 删除的非等形 view，可重新评估该变体；验证必须记录专用 pattern 命中，而不是只看配置或总命中。
- 若只需 W8A8 性能优化，应优先分析 direct 融合覆盖不到的图节点，而不是继续扩大一个当前无独立命中的开关。

## 8. 原始结果与 Patch

原始日志和结果：

```text
server-results/inductor-feature/20261122_140509/dynamic_norm_quant_qwen27b/
server-results/inductor-feature/20261122_140509/dynamic_norm_quant_qwen06b/
```

Patch：

```text
doc/InfraDocs/TorchNPU_Inductor/feature/Patch/DynamicNormQuant/0002-cover-dynamic-norm-quant-flag-forwarding.patch
```

应用：

```bash
git apply Patch/DynamicNormQuant/0002-cover-dynamic-norm-quant-flag-forwarding.patch
```

回退本阶段新增测试：

```bash
git revert 7fc0fed88421587b65b1275920b0a2273c045f95
```

本阶段没有修改融合实现，因此回退测试不会改变 direct 或 view pattern 的默认行为。

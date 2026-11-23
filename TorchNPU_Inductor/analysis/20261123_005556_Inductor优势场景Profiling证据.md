# Inductor 相对 Default 的优势场景与因果证据

日期：2026-11-23。环境：A5-25，Ascend950DT，torch 2.13 / vLLM 0.33。

## 1. 结论：Inductor 的收益分为“直接执行收益”和“持续负载放大”

这组实验支持下面的因果链：

```text
Inductor 做图融合和数据移动消除
        ↓
减少独立算子、数据重排/拷贝和部分 Host 调用
        ↓
prefill/decode 的 Computing 持平或小幅下降，Free（设备间隙）明显下降
        ↓
单请求或稳定 batch 的 TTFT、TPOT 降低——这是直接执行收益
        ↓
持续请求到达时，请求更快完成，排队和长间隔停顿减少
        ↓
TTFT、平均 TPOT 和吞吐出现更大的端到端收益——这是持续负载放大
```

关键是：**单次 kernel 没有大幅提速，不等于端到端没有明显收益；端到端的大幅收益也不能全部当成 kernel 提速。**

| 场景 | 直接执行证据 | 持续负载下的最终收益 |
|---|---|---|
| 8B-W8A8 | QPS4 Profile：Stage/步 **-3.54%**，Free/步 **-20.14%** | QPS16 吞吐 **+111.26%**，TPOT **-53.84%** |
| 27B-W8A8 | 受控 decode：Stage **-1.48%**，Computing **-0.09%**，Free **-27.82%** | QPS4 两轮 TPOT **-16.80%/-20.39%**，TTFT **-79.41%/-85.43%** |
| 30B-A3B-W8A8 | 单请求 prefill Stage **-25.96%**；decode Stage/步 **-5.78%** | QPS1 TTFT **-24.60%**、TPOT **-6.35%**；QPS16 吞吐 **+11.63%** |

下文按“原始收益 → 直接执行原因 → 持续负载为何放大”的顺序展开。

## 2. 指标和实验方法

### 2.1 指标分别回答什么问题

- **TTFT**：从请求到收到首 token 的时间，主要反映 prefill 以及首 token 前的等待。
- **TPOT**：请求级平均输出间隔；**ITL**：相邻 token 的间隔分布。27B 两轮原始数据中二者均值一致，因此可用 ITL 总量分解解释平均 TPOT 收益。平均 TPOT 大幅变化而 median ITL 变化很小，说明变化主要来自长尾间隔，而不是普通间隔普遍同比例变化。
- **Computing**：Profile 窗口内的设备计算时间。
- **Free**：设备执行间隙。
- **Stage**：该 Profile 窗口的总阶段时间，包括计算、通信和间隙。
- **Host launch 次数/时间**：Host 侧提交工作的调用次数和累计耗时。它是 Host API 统计，不属于 `Stage = Computing + 非重叠通信 + Free` 的设备时间分解，不能与 Free 的减少量直接相加。

### 2.2 两条路径是可比的编译路径，不是编译与 eager 的对比

- Default：`AscendCompiler`；
- Inductor：`backend=inductor`，底层为 `triton_experimental`；
- 两者均为 `O2`、`FULL_AND_PIECEWISE`，均启用 ACL graph replay。

模型、TP、seed、prefix caching、最大序列数和显存参数保持一致。backend 自动带出的编译选项属于整条路径差异，报告不把收益归因于某个孤立开关。

### 2.3 原始性能与 Profile 各自承担的证据角色

- 原始性能使用未开启 Profiler 的 `vllm bench serve`：每组 200 条正式请求全部成功，输入长度和输出长度逐请求一致，每条输出 256 tokens。
- ShareGPT Profile 在正式 benchmark 的第 5 秒和第 30 秒各采集 2 秒；首个窗口缺少完整 TASK 数据，报告使用第二个完整窗口。
- 补充实验使用固定 token ID 和固定输入长度，分别采集 prefill 与 decode。27B decode 两边均为相同 64 个引擎步；30B prefill 在收到首 token 时停止采集。
- 算子累计时间不等于端到端时间。ShareGPT 窗口按实际引擎步数归一化；受控实验优先使用相同步数做直接比较。
- Profile 会扰动调度，因此最终收益数字以非 Profile benchmark 为准，Profile 用于解释执行机制。

## 3. 场景一：8B-W8A8——小幅直接收益，在高并发下转化为大幅吞吐收益

### 3.1 先看最终收益

| 负载 | 指标 | Default | Inductor | 变化 |
|---|---|---:|---:|---:|
| QPS4 | 输出吞吐 | 961.320 | 989.916 | +2.97% |
| QPS4 | 平均 TPOT | 11.632 ms | 7.092 ms | -39.03% |
| QPS4 | 平均 TTFT | 86.133 ms | 37.059 ms | -56.98% |
| QPS16 | 输出吞吐 | 1433.186 | 3027.748 | **+111.26%** |
| QPS16 | 平均 TPOT | 19.796 ms | 9.138 ms | **-53.84%** |
| QPS16 | 平均 TTFT | 9028.053 ms | 1289.953 ms | **-85.71%** |

这些数字说明端到端收益确实存在，但还需要拆分直接执行收益与持续负载放大。

### 3.2 图层变化：融合 RMSNorm，并消除部分 split/view 和 Slice

8B 生成代码包含：

- `triton_unk_fused_npu_rms_norm_1`：在一个 kernel 内完成 BF16→FP32、平方、均值、rsqrt、权重乘和输出转换；
- `triton_unk_fused_npu_rms_norm_split_with_sizes_view_2`：从 QKV 拼接张量按 Q 的偏移直接读取并完成归一化；
- `triton_unk_fused_npu_rms_norm_split_with_sizes_view_3`：按 K 的偏移直接读取并完成归一化。

这些 kernel 的来源节点同时包含 `split_with_sizes`、`view` 和 `npu_rms_norm`。也就是说，Inductor 直接对拼接张量寻址，省去独立的数据重排/拷贝节点。

Profile 中对应现象为：

- 原生 `RmsNorm`：15,622 次 → 0；
- `Slice`：23,540 次、76.065 ms → 8,626 次、24.840 ms；
- 新增若干 RMSNorm/split/view Triton kernel。

新增 Triton kernel 并非每个都比原生 RMSNorm 更快。上述变化直接证明了**中间数据移动和独立算子调用减少**；它们与后面的 Host/Free 变化共同支持融合降低执行开销的解释，而不是证明某个 RMSNorm 微 kernel 单独大幅领先。

### 3.3 稳定执行时的直接收益：开销和间隙下降

QPS4 第二个 Profile 窗口按引擎步归一化：

| 指标 | Default | Inductor | 变化 |
|---|---:|---:|---:|
| 算子数/步 | 375.72 | 339.05 | **-9.76%** |
| Host launch 次数/步 | 58.96 | 54.11 | **-8.22%** |
| Host launch 时间/步 | 331.86 μs | 323.11 μs | -2.63% |
| Computing/步 | 4755.38 μs | 4761.98 μs | +0.14%，持平 |
| Free/步 | 1054.04 μs | 841.73 μs | **-20.14%** |
| Stage/步 | 5809.42 μs | 5603.71 μs | **-3.54%** |

这里 Computing 没有下降，Stage/步的下降在时间线上直接体现为 **Free/步减少，并伴随 Host launch 次数和时间下降**。Stage/步下降 3.54%，与 QPS4 前 50 个请求的 TPOT 从 7.264 降至 7.001 ms（约 -3.6%）相吻合。因此，稳定执行阶段的直接收益在时间线上体现为**设备间隙和 Host 工作减少**；融合及 Slice 减少与这一解释一致，而主要计算 kernel 没有提速。

### 3.4 持续高并发下为何收益变大

QPS16 的 Default 实际请求吞吐只有 5.598 req/s，明显低于 16 req/s 的到达速率，因此请求持续积压；Inductor 达到 11.827 req/s。最大并发请求数从 150 降到 74。

直接执行收益让请求更早完成，从而降低排队和长间隔停顿，最终形成 QPS16 的 2.11 倍输出吞吐。**2.11 倍是系统级结果，不是某个 kernel 的 2.11 倍加速。**

## 4. 场景二：27B-W8A8——可重复的 QPS4 收益，TPOT 与 TTFT 分别归因

### 4.1 两轮结果均复现

| 轮次 | 指标 | Default | Inductor | 变化 |
|---|---|---:|---:|---:|
| 1 | 输出吞吐 | 892.248 | 937.653 | +5.09% |
| 1 | 平均 TPOT | 29.016 ms | 24.140 ms | -16.80% |
| 1 | 平均 TTFT | 1887.007 ms | 388.489 ms | -79.41% |
| 2 | 输出吞吐 | 876.912 | 937.780 | +6.94% |
| 2 | 平均 TPOT | 29.655 ms | 23.607 ms | -20.39% |
| 2 | 平均 TTFT | 2144.940 ms | 312.448 ms | -85.43% |

输入输出长度相同，200 条请求均成功。这是重复性最好的场景。

### 4.2 TTFT：主要是排队减少，不是 prefill 直接提速数倍

按请求顺序每 50 个划分一段：

| 轮次与路径 | 1～50 | 51～100 | 101～150 | 151～200 |
|---|---:|---:|---:|---:|
| 第 1 轮 Default | 146 ms | 1745 ms | 3407 ms | 2250 ms |
| 第 1 轮 Inductor | 120 ms | 358 ms | 823 ms | 253 ms |
| 第 2 轮 Default | 141 ms | 1767 ms | 3703 ms | 2969 ms |
| 第 2 轮 Inductor | 123 ms | 281 ms | 598 ms | 248 ms |

Default 在中段出现秒级积压，Inductor 的峰值更低且更早回落。作为低负载参照，同一模型 QPS1 的 TTFT 加速比只有 1.12～1.15。

所以，QPS4 的 4.86～6.87 倍 TTFT 加速主要来自**首 token 前等待缩短**，不能解释为 prefill kernel 直接提速数倍。

### 4.3 TPOT：93%～95% 的收益来自长尾间隔减少

每轮每条路径都有 51,000 个 ITL。以 50 ms 为界拆分：

| 轮次 | 平均 TPOT 总收益 | `>50 ms` 长尾贡献 | `≤50 ms` 间隔贡献 |
|---|---:|---:|---:|
| 1 | 4.876 ms | **4.551 ms，93.35%** | 0.324 ms，6.65% |
| 2 | 6.048 ms | **5.772 ms，95.44%** | 0.276 ms，4.56% |

长尾绝对耗时分别减少 232.124 s 和 294.394 s。长尾只占 ITL 样本数约 8%，因此对均值影响很大，但对中位数影响有限。这正好解释了：

- 平均 TPOT 降低 16.80%/20.39%；
- median ITL 仅降低约 1.3%/1.5%。

因此，**平均 TPOT 的大幅收益主要来自长尾停顿减少，而不是普通 decode 间隔普遍同比例提速。**

### 4.4 普通 decode 步的直接收益：来自 Host/设备间隙，而非大 kernel

受控实验固定为 256-token 输入、8 并发、相同 token ID；Default 和 Inductor 均采集相同 64 个 decode 引擎步：

| 指标 | Default | Inductor | 变化 |
|---|---:|---:|---:|
| Computing | 681.981 ms | 681.362 ms | -0.09%，持平 |
| `QuantBatchMatmulV3` | 322.089 ms | 321.647 ms | -0.14%，持平 |
| HCCL | 338.328 ms | 337.999 ms | -0.10%，持平 |
| Host launch 时间 | 109.735 ms | 103.036 ms | **-6.10%** |
| Free | 46.311 ms | 33.428 ms | **-27.82%** |
| Stage | 839.999 ms | 827.540 ms | **-1.48%** |

Computing、GEMM 和 HCCL 都没有实质变化，下降的是 Host launch 时间和 Free。因此，普通 decode 步的小幅直接收益在时间线上**直接体现为 Host launch 时间下降和 Free 缩短**，而不是大 kernel 计算提速。

这里要区分两个层次：受控实验解释普通步的小幅收益；原始 ITL 分解证明 QPS4 平均 TPOT 的大幅收益主要来自长尾减少。两者的比例不能互相代替。

## 5. 场景三：30B-A3B-W8A8——分别证明 prefill 和 decode 的直接收益

### 5.1 原始性能

| 负载 | 指标 | Default | Inductor | 变化 |
|---|---|---:|---:|---:|
| QPS1 | 平均 TPOT | 9.094 ms | 8.516 ms | **-6.35%** |
| QPS1 | 平均 TTFT | 75.735 ms | 57.107 ms | **-24.60%** |
| QPS4 | 平均 TPOT | 16.940 ms | 14.871 ms | -12.21% |
| QPS16 | 输出吞吐 | 1450.127 | 1618.730 | **+11.63%** |
| QPS16 | 平均 TPOT | 19.822 ms | 17.738 ms | -10.51% |
| QPS16 | 平均 TTFT | 8939.095 ms | 7312.354 ms | -18.20% |

QPS1 输出吞吐仅变化 +0.09%，因为固定到达率限制了总吞吐，不影响延迟收益的判断。

在双方到达并发均不超过 1 的 30 个共同请求中，TTFT 为 66.766 → 50.722 ms，TPOT 为 7.235 → 6.953 ms。QPS1 的直接收益不需要用排队解释。

### 5.2 Prefill：Stage/TTFT 缩短几乎全部体现为 Free 减少

独立实验使用单请求、固定 256-token 输入和相同 token ID，并在收到首 token 时停止 Profile：

| 指标 | Default | Inductor | 变化 |
|---|---:|---:|---:|
| TTFT | 97.452 ms | 70.369 ms | **-27.79%** |
| Prefill Stage | 101.746 ms | 75.331 ms | **-25.96%** |
| Computing | 30.373 ms | 29.736 ms | -2.10% |
| Free | 71.373 ms | 45.595 ms | **-36.12%** |
| Host launch 次数 | 1374 | 1133 | **-17.54%** |
| Host launch 时间 | 6.247 ms | 4.835 ms | **-22.60%** |

Stage 减少 26.415 ms，解释 TTFT 减少量的 97.53%；Free 减少占 Stage 减少量的 97.58%。主要 MoE/GEMM/Attention 算子累计时间基本持平，例如 `GroupedMatmulSwigluQuantV2` 仅下降 0.83%，Attention 仅下降 0.36%。

因此，**30B QPS1 的 prefill Stage/TTFT 缩短几乎全部体现为 Free 减少**；Host launch 次数和时间同步下降，为调用开销减少提供旁证。受控形状上的 27.79% 与原 ShareGPT QPS1 的 24.60% 属于同量级直接收益。

### 5.3 Decode：Computing 小幅下降，Free 明显缩短

QPS1 ShareGPT 的第二个有效 Profile 窗口按引擎步归一化：

| 指标 | Default | Inductor | 变化 |
|---|---:|---:|---:|
| 算子数/步 | 861.02 | 790.90 | **-8.14%** |
| Host launch 次数/步 | 67.10 | 65.52 | -2.35% |
| Host launch 时间/步 | 513.77 μs | 505.48 μs | -1.61% |
| Computing/步 | 6585.18 μs | 6366.61 μs | **-3.32%** |
| Free/步 | 681.06 μs | 479.89 μs | **-29.54%** |
| Stage/步 | 7266.24 μs | 6846.50 μs | **-5.78%** |

Stage/步下降 5.78%，与原始 QPS1 TPOT 下降 6.35% 接近。算子侧还出现原生 `RmsNorm` 消失、`Slice` 从 31,828 次降为 10,600 次，以及 `triton_unk_fused_npu_rms_norm_split_with_sizes_view_*` 融合 kernel。

所以 decode TPOT 的直接变化是：**Computing 小幅下降、Free 明显缩短**；融合和 Slice 减少与这一变化同时出现，为执行开销下降提供图层证据。

### 5.4 完整请求根因：两条具体替换链减少 prefill Host 工作

为了定位 NPU 未忙碌时间为什么减少，进一步对同一 30B 单请求的完整 64-token trace 分析 Host 事件，并以第二个 `MODEL_EXECUTE` 划分 prefill/首 token 与后续 decode。Free 事件标记的时间分解为：

| 阶段 | Default Free | Inductor Free | 变化 |
|---|---:|---:|---:|
| Prefill/首 token | 80.942 ms | 47.542 ms | **-33.399 ms** |
| 后续 decode | 69.753 ms | 69.429 ms | -0.323 ms |

这次完整请求的客户端时间为 561.230 → 528.440 ms，净减少 32.790 ms；TTFT 为 106.950 → 73.360 ms，减少 33.590 ms，首 token 之后反而增加 0.799 ms。因此，净收益来自首 token 前，而不是 63 次 decode 图重放普遍变快。

将 48 个模型层按相同的 Host 锚点对齐，可以得到下表。每段墙钟时间只计算一次；Free 是其中 NPU 没有执行相关任务的部分。

| 模型层阶段 | Default 墙钟 | Inductor 墙钟 | 墙钟节省 | 其中 Free 节省 |
|---|---:|---:|---:|---:|
| QKV + Q/K RMSNorm | 14.342 ms | 3.271 ms | **11.071 ms** | 10.970 ms |
| RoPE | 7.786 ms | 5.704 ms | **2.082 ms** | 2.577 ms |
| Attention 准备 + Attention | 15.794 ms | 12.425 ms | **3.369 ms** | 3.362 ms |
| 输出投影 + 残差归一化 | 7.802 ms | 6.902 ms | **0.900 ms** | 0.784 ms |
| MoE auto-functionalization 与入口准备 | 12.332 ms | 0.672 ms | **11.660 ms** | 11.622 ms |
| MoE 自定义算子内部 | 22.725 ms | 19.457 ms | **3.269 ms** | 2.352 ms |
| **六段合计** | **80.781 ms** | **48.430 ms** | **32.351 ms** | **31.667 ms** |

六段合计的 NPU 忙碌时间只减少 0.686 ms，而 Free 减少 31.667 ms。剩余 1.733 ms Free 收益位于上述层内阶段边界之外。因此，收益的来源已经可以直接定位为 **Host 图运行包装、算子准备和提交间隔减少**，不是主要计算 kernel 普遍提速。

#### 替换链一：CANN RMSNorm Host 调用链 → Triton RMSNorm + 地址计算

Default 在 Q/K 归一化中执行 `split_with_sizes/view + npu::npu_rms_norm`。Inductor 将 Q、K 分别生成为两个 Triton reduction kernel，直接从原 QKV 张量按 stride/offset 读取，再用 `reinterpret_tensor` 构造下游视图。RoPE、`npu_add_rms_norm_bias` 和 `npu_add_rms_norm_quant` 仍然保留，没有计为被替换项。

- Default 的 97 次 `npu::npu_rms_norm` Host 外层调用累计 9.927 ms，Inductor 为 0；其中 9.372 ms 与 NPU Free 直接重叠。
- CANN `RmsNormGetWorkspaceSize` 和 `Contiguous` 是该外层调用的内部路径，不重复计时。
- 两条路径的 Q/K RMSNorm device 任务都为 194 个；累计 device 时间仅从 0.913 ms 降为 0.894 ms，节省 0.019 ms。

所以，**RMSNorm 数学计算仍然执行，device 计算也基本相同；真正省掉的是每次独立 CANN 调用的 Host workspace、参数准备、dispatch 以及独立 split/view 节点。** 这解释了 QKV + Q/K RMSNorm 段 11.071 ms 墙钟收益的主要部分。

#### 替换链二：MoE auto-functionalization 与 clone → 直接自定义算子调用

Default 的 `vllm.moe_forward` 声明了 `mutates_args=["hidden_states"]`。带 Python 栈的独立 prefill Profile 显示，其 piecewise AOT 运行路径在每一层调用该算子前执行：

```text
auto_functionalized_v2_dense
  → _generate_new_op_kwargs_from_bases
  → maybe_copy
  → clone_preserve_strides
  → aten::clone
```

48 个根级 MoE 前 clone 全部属于这条调用链。原完整请求中，`aten::view → aten::as_strided` 之间的 48 段无 profiler 外层事件间隔合计 9.425 ms，带栈 Profile 将其定位为上述参数重建过程；此外还发生 48 次 `clone_preserve_strides`。Inductor 生成代码直接执行 `torch.ops.vllm.moe_forward.default(buf5, buf5, ...)`，不再经过运行时 `auto_functionalized_v2` 包装，原 trace 中这 48 次根级 clone 为 0。

因此，**MoE 的数学计算没有被删除，`moe_forward` 仍执行 48 次；被删除的是它外面的 Python 参数重建、可写参数复制和相关 ATen 调度。** 这一段墙钟从 12.332 ms 降为 0.672 ms，是第二条主要收益来源。带栈采集只用于确认调用链，其自身较大的 profiler 开销不用于性能差值计算。

最终因果链为：

```text
RMSNorm 改用 Triton 实现并折叠 split/view
+ MoE 入口移除 auto-functionalization 参数重建和 clone
+ 其他 ATen view/alias/empty 与 Python 调度减少
→ Host 在关键路径上的工作减少
→ prefill Free 减少 33.399 ms
→ TTFT 减少 33.590 ms
→ 完整请求净减少 32.790 ms
```

阶段数据见 `../../../../profilling/20261123_032729_inductor_root_cause_evidence/20261123_032729_30b_prefill_stage_summary.csv`；MoE Python 栈见同目录 `20261123_032729_30b_moe_auto_functionalization_stack.json`；Q/K RMSNorm device 对照见 `20261123_032729_30b_qk_norm_device_summary.json`。真实事件图仍为 `20261123_032729_inductor_root_cause_evidence/20261123_032729_30b_removed_host_path_timeline.png`。双方从零开始、保留每个函数/kernel 的完整对齐视图见 `../../../../profilling/20261123_023654_inductor_visual_evidence/20261123_023654_30b_aligned_event_timeline.html`；静态总览和首层放大图仍保存在 `20261123_023654_inductor_visual_evidence/` 图片目录。

### 5.5 高并发扩展

QPS16 下最大并发请求数从 149 降为 137，输出吞吐提高 11.63%。双方都存在积压，因此高负载 TTFT 收益也包含等待减少；它不应全部归因于 prefill 或 decode kernel。

## 6. 共性结论与适用边界

### 6.1 三个场景共同支持的结论

1. Inductor 的实际作用点是图融合、减少中间数据移动和部分调用开销。
2. 直接执行收益可以出现在 prefill 或 decode，主要表现为 Host/Free 下降；Computing 可能持平，也可能小幅下降。
3. 在持续请求负载下，直接执行收益还会降低排队和长尾停顿，因此端到端倍率可以远大于单步收益。
4. 没有证据表明主要 GEMM、Attention、MoE 或 HCCL kernel 被普遍大幅提速。

### 6.2 长上下文不能外推

已有 27B、32768/1024、8 并发、TP2 的 Profile 给出反例：

- decode 的 `QuantBatchMatmulV3` 为 646.380 → 645.774 ms，`FusedInferAttentionScore` 为 1304.604 → 1304.585 ms，基本持平；
- 六类小算子的消失或减少合计节省约 120.393 ms；
- 新增 10 类 Triton kernel 累计 159.235 ms，全部算子累计时间反而增加 0.743%，Stage 增加 1.108%；
- Host launch 时间下降 6.905%，但次数不变，也不足以抵消设备端回退；
- 旧 prefill 还存在 RMSNorm 相关生成 kernel 的 launch/tiling 异常。

因此，本报告的结论限定于已经验证的模型和中短上下文负载，不能外推为长上下文或所有模型普遍受益。

### 6.3 性能结论不代表精度结论

这些场景的固定输入 logprob 检查均未通过原矩阵容差。本报告只证明性能及执行机制，不声称数值一致性或精度优势。

## 7. 证据文件与复现材料

原始证据位于 `../../../../profilling/20261123_005556_inductor_advantage_evidence/`，文件名均带 `20261123_005556` 前缀：

- `20261123_005556_evidence_manifest.json`：场景、原始批次、Profile 路径和限制；
- `*_request_metrics_summary.csv`：原始请求指标；
- `*_ttft_quarters.csv`：TTFT 随请求进程变化；
- `*_raw.json`、`*_compare.json`：逐请求结果和指标对比；
- `*_server.json`：实际 backend、图模式和服务参数；
- `8b_w8a8_output_code_*.py`：8B Inductor 生成代码；
- `*_normalized_operators.csv`：按引擎步归一化的算子对比；
- `*_timeline_summary.csv`：Computing、Free、Stage、launch 汇总；
- `27b_w8a8_32k_*`：长上下文反向边界；
- `20261123_005556_profile_sharegpt_qps.py`、`20261123_005556_compare_timed_profiles.py`：采集和分析脚本快照。

补充因果验证位于 `../../../../profilling/20261123_014925_inductor_causal_closure_evidence/`：

- `20261123_014925_closure_manifest.json`：补充实验与结果索引；
- `20261123_014925_27b_qps4_tpot_decomposition.csv`：27B 长尾贡献分解；
- `20261123_014925_27b_fixed8_256_decode_summary.csv`：27B 受控 decode；
- `20261123_014925_30b_fixed1_256_prefill_summary.csv`：30B 单请求 prefill；
- 同目录还保存对应原始 Profile CSV、配置和采集脚本。

原始二进制 trace 保留在 A5 服务器结果目录；本地归档保留可分析的 CSV、JSON、生成代码和脚本，避免复制大体积 trace。

## 8. 请求级可视化

请求内部的完整时间分解、真实 NPU 忙碌度时间轴和多请求生命周期图见 `20261123_023654_请求级可视化证据.md`。这些图直接使用请求时间戳和 kernel 事件，不再以汇总百分比代替请求执行过程。

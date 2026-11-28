# Inductor 性能矩阵、收益来源与最新 Profile 优化分析

日期：2026-11-26。环境：A5-25，Ascend950DT，torch 2.13、vLLM/vllm-ascend 0.33。

本文整合三部分内容：

1. 14 个完整 TP1 模型在捕获 1024 下的性能矩阵；
2. Inductor 相对 default 加速比不明显的原因；
3. 30B、27B、VL 的最新相同形状 Profile、生成代码证据和后续优化方向。

本文只讨论性能。固定输入 token/logprob 精度需要单独判断，不能由性能结果推导。

## 1. 一页结论

1. **当前 default 不是未编译的 eager。** default 和 Inductor 都使用 O2、`FULL_AND_PIECEWISE` 和最大 1024 的图捕获。default 通过扩大图捕获已经获得了一部分 Host 提交和设备空闲收益，因此 Inductor 的相对优势小于捕获 32 时。
2. **Inductor 的主要收益来自小算子融合、数据准备减少和执行间隙缩短。** 现有有效对照没有证明 GEMM、Attention、MoE、HCCL 等主要计算普遍大幅提速。
3. **低 QPS 的吞吐受到请求到达率限制。** QPS1/QPS4 下吞吐接近 1.000 不能单独用于判断编译器是否有效，应同时看 TTFT、TPOT 和固定形状执行时间。
4. **当前 84 个“模型 × 负载”组合中，63 个吞吐加速比中位数在 0.98～1.02。** 但 32K 输入有 7/14 个模型吞吐下降超过 2%，说明长输入存在需要单独处理的真实回退。
5. **30B 的 decode 收益真实且稳定，但 MoE 回退抵消了部分收益。** Norm/Slice 融合降低 Computing；两类 GroupedMatmul 在相同形状和调用次数下仍慢约 4%。
6. **27B 长 prefill 的主要问题是输入打包、Sigmoid+Mul 和状态清零。** 主要 Attention 和 QuantBatchMatmul 基本持平或略快，不是本次回退的主要来源。
7. **VL 长 prefill 的主要问题是 MRoPE 后的完整 QKV 读改写。** 不需要变化的 V 部分也被读取和写回，是最明确的复制优化对象。
8. **下一步应优先修复具体生成代码和布局问题，而不是继续扩大长 prefill 图捕获或全局开启 Coordinate Descent。**

## 2. 比较对象与指标定义

### 2.1 两条执行路径

- **default**：vLLM Ascend 原有的 `AscendCompiler` 编译路径。
- **Inductor**：`compilation_config.backend=inductor`，底层 NPU backend 为 `triton_experimental`。

当前正式性能矩阵中，两条路径均使用：

- O2 编译；
- `FULL_AND_PIECEWISE`；
- `max_cudagraph_capture_size=1024`；
- 捕获尺寸 `[1,2,4,8,16,24,32,64,128,256,512,1024]`；
- 相同模型、量化方式、TP、输入和请求参数。

### 2.2 性能指标

本文加速比均按“越高越好”计算：

```text
吞吐加速比 = Inductor 输出吞吐 / default 输出吞吐
TPOT 加速比 = default 平均 TPOT / Inductor 平均 TPOT
TTFT 加速比 = default 平均 TTFT / Inductor 平均 TTFT
```

因此：

- `1.00` 表示基本持平；
- `1.03` 表示 Inductor 约好 3%；
- `0.95` 表示 Inductor 约慢 5%。

延迟指标含义：

- **TTFT**：从请求发出到收到第一个 token 的时间，包含排队和 prefill。
- **TPOT**：首个 token 之后，平均每个输出 token 的时间间隔。
- **ITL**：相邻两个输出 token 的实际间隔；median ITL 更接近普通 decode 的稳定速度。
- **Prefill**：处理输入 prompt 的阶段。
- **Decode**：逐个生成输出 token 的阶段。

### 2.3 图编译与图回放不是同一概念

两条路径都经过图编译，也都可能执行 ACL graph replay：

- **FULL**：整个可捕获执行单元进入一张图，常用于纯 decode。
- **PIECEWISE**：模型按图片段回放，Attention 等不兼容操作可留在图外。
- **NONE**：当前形状没有匹配的捕获图，直接执行编译后的 callable。

`max_cudagraph_capture_size=1024` 只表示运行时 token 数不超过 1024 时有机会命中捕获图，不表示任意长度的 prefill 都被回放。超过上限时会回到 `NONE`。

## 3. 数据与验证范围

### 3.1 正式性能矩阵

14 个完整 TP1 模型均完成：

- 3 轮 × 6 种负载；
- 每轮 default 与 Inductor 独立服务；
- 18 份 compare JSON；
- 3 份 Inductor evidence；
- 两条路径的 server.json 均记录捕获上限 1024。

六种负载为：

| 负载 | 请求特点 |
|---|---|
| ShareGPT QPS1 | 200 条真实输入，固定输出 256 tokens，1 请求/秒 |
| ShareGPT QPS4 | 同上，4 请求/秒 |
| ShareGPT QPS16 | 同上，16 请求/秒 |
| ShareGPT QPS∞ | 200 条请求尽快到达 |
| 随机 8192/1024 | 输入 8192、输出 1024 tokens，最大并发 8 |
| 随机 32768/1024 | 输入 32768、输出 1024 tokens，最大并发 8 |

每个服务在正式测量前执行 warmup；服务启动、冷编译和图捕获时间不计入在线服务吞吐。冷启动成本应另行评价。

正式结果见：

- [20261126_perf_cap1024_SUMMARY.md](../tests/20261126_perf_cap1024_SUMMARY.md)
- `server-results/benchmark/20261125_perf_cap1024_01/`

### 3.2 最新相同形状 Profile

| 场景 | 固定输入 | 采集窗口 |
|---|---|---|
| Qwen3-30B-A3B-W8A8 | 8 请求 × 256 输入 tokens | prefill；第 64～128 输出 token 的纯 decode |
| Qwen3.8-27B-W8A8 | 8 请求 × 32768 输入 tokens | 长 prefill；第 128～256 输出 token 的 decode |
| Qwen2.5-VL-7B-Instruct | 8 请求 × 32768 输入 tokens | 长 prefill；第 128～256 输出 token 的 decode |

控制条件：

- default 和 Inductor 使用完全相同的 token ID；
- TP1、相同 NPU、相同并发和输出长度；
- 两边均为 1024 捕获；
- 复用当前矩阵的编译缓存；
- 每组 default/Inductor 均成功，共 6 次完整运行。

Profile 会扰动服务性能。本文只使用对齐后的设备窗口和算子数据解释原因，不使用 Profile 状态下的客户端吞吐替代正式结果。

## 4. 捕获 1024 矩阵的整体表现

84 个组合中：

- 63 个吞吐加速比中位数位于 0.98～1.02；
- 2 个组合的吞吐中位数超过 1.05；
- 32K 输入没有模型取得超过 2% 的吞吐提升，7 个模型下降超过 2%。

按相同负载统计 14 个模型：

| 负载 | 提升超过 2% | 下降超过 2% | 主要特征 |
|---|---:|---:|---|
| ShareGPT QPS1 | 0/14 | 0/14 | 吞吐几乎全部持平 |
| ShareGPT QPS4 | 0/14 | 0/14 | 吞吐几乎全部持平 |
| ShareGPT QPS16 | 3/14 | 1/14 | 部分模型出现小幅收益 |
| ShareGPT QPS∞ | 3/14 | 0/14 | 饱和负载下有少量正收益 |
| 随机 8192/1024 | 4/14 | 3/14 | 模型间分化明显 |
| 随机 32768/1024 | 0/14 | 7/14 | 长输入回退最集中 |

TTFT 也呈现长输入回退：8K 和 32K 输入下均有 9/14 个模型下降超过 2%。

代表性结果：

| 模型与负载 | 吞吐加速比中位数 | 说明 |
|---|---:|---|
| 30B-W8A8，QPS16 | 1.0324 | 三轮稳定正收益 |
| 30B-W8A8，QPS∞ | 1.0515 | 饱和负载下约 5% |
| 8B-W8A8，QPS16 | 1.0093 | 约 1% |
| 27B-W8A8，QPS16 | 1.0006 | 基本持平 |
| 27B-W8A8，32K | 0.9539 | 稳定回退 |
| Qwen2.5-VL，32K | 0.9288 | 明显回退 |
| Qwen3.5-0.8B，QPS16 | 0.9758 | 轮间差异较大 |

## 5. 为什么整体加速比不明显

### 5.1 default 的图捕获与 Inductor 收益重叠

Inductor 在当前模型上的主要价值包括：

- 融合 RMSNorm、Add、Sigmoid、Mul 等小算子；
- 减少 Split、View、Slice 和数据搬运；
- 减少 Host 参数准备和 kernel 提交；
- 缩短设备空闲与请求之间的执行间隙。

扩大 default 的图捕获也能减少其中一部分 Host 和空闲开销。

此前 `{default, Inductor} × {捕获 32, 捕获 1024}` 的 QPS16 对照为：

| 模型 | Inductor/default，捕获 32 | Inductor/default，捕获 1024 |
|---|---:|---:|
| 8B-W8A8 | +10.3% | +1.7% |
| 30B-A3B-W8A8 | +14.7% | +3.3% |
| 27B-W8A8 | +9.3% | 约 +1.4% |

这说明两种优化不能完全叠加。当前 1024 矩阵中的较小 Inductor 净收益，不表示 Inductor 在捕获 32 时也没有价值。

详细数据见 [20261124_Inductor性能收益来源与捕获窗口对照.md](20261124_Inductor性能收益来源与捕获窗口对照.md)。

### 5.2 主要计算没有普遍变快

已有相同形状 Profile 多次显示：

- GEMM、QuantBatchMatmul 基本持平；
- Attention 基本持平；
- MoE 数学计算仍在执行，没有被融合操作删除；
- HCCL 没有因为 Inductor 普遍提速。

例如早期 27B 纯 prefill 对照中，default32/default1024/Inductor32 的 Computing 分别为 279.1/279.7/282.7 ms，主要差异在 Free 和 Host launch，而不是 Computing。

因此即使小算子或 Host 路径改善较大，只要主要大算子占据大部分时间，端到端收益仍然有限。

### 5.3 低 QPS 吞吐受到达率限制

ShareGPT 每条请求固定输出 256 tokens。当 QPS1 且服务能够及时处理请求时：

```text
输出吞吐 ≈ 1 请求/秒 × 256 tokens = 256 tokens/秒
```

即使单次计算更快，只要请求仍按 1 QPS 到达，总吞吐也不会明显增加。

因此：

- QPS1/QPS4 的吞吐接近 1.000 是预期现象；
- 这些负载应重点看 TTFT、TPOT 和固定形状执行时间；
- QPS16、QPS∞ 更适合观察系统吞吐差异。

### 5.4 端到端收益经常来自排队与长尾

平均 TPOT 可以拆成：

```text
平均 TPOT ≈ median ITL + prefill/调度干扰项
```

新请求的 prefill 可能阻塞正在 decode 的请求，形成跨请求共同停顿。Inductor 缩短 prefill 或执行衔接后，可能出现：

- median ITL 基本不变；
- P95/P99 ITL 下降；
- 超长间隔数量减少；
- 请求积压更快消退；
- 平均 TPOT 和 TTFT 明显改善。

这类收益属于系统执行改善，不能写成 decode 大算子同幅度提速。

### 5.5 小模型结果更容易受到运行波动影响

Qwen3.5-0.8B 的 QPS16 三轮吞吐加速比为：

```text
0.9638 / 0.9758 / 1.0030
```

Inductor 三轮都消除了 `>50 ms` 长间隔，但普通 median ITL 有时反而更慢。说明长尾改善和普通 kernel 回退可以同时存在。

对 1%～2% 的差异，应增加重复次数并结合相同形状 Profile，不宜只根据一轮结果下结论。

## 6. 30B：Norm/Slice 融合有效，MoE 抵消部分收益

### 6.1 非 Profile 结果

30B-W8A8 的 QPS16 三轮吞吐加速比为：

```text
1.0334 / 1.0323 / 1.0324
```

median ITL 从约 15.64 ms 降至 15.10 ms。说明约 3% 的收益不仅来自排队减少，普通 decode 间隔也确实改善。

### 6.2 固定纯 decode 窗口

| 指标 | default | Inductor | 变化 |
|---|---:|---:|---:|
| Stage | 668.36 ms | 651.44 ms | -16.93 ms |
| Computing | 651.80 ms | 638.01 ms | -13.80 ms |
| Free | 16.56 ms | 13.43 ms | -3.13 ms |
| FULL graph replay | 64 次 | 64 次 | 相同 |
| Host launch 次数 | 4482 | 4482 | 相同 |

双方 replay 和 launch 次数相同，说明收益不是简单的图回放次数差异，而主要体现在 Computing 下降。

### 6.3 收益来源

| 算子变化 | 时间影响 |
|---|---:|
| 原生 RmsNorm 被消除 | -32.57 ms |
| Slice：28.17 → 8.12 ms | -20.05 ms |
| 新增主要 RMSNorm 融合 Triton kernel | +24.55 ms |
| QuantBatchMatmulV3：69.51 → 69.13 ms | 基本持平 |

Norm/Slice 融合的净效果为正，是实际收益来源。

### 6.4 MoE 回退

| MoE 算子 | default | Inductor | 增量 |
|---|---:|---:|---:|
| GroupedMatmulSwigluQuantV2 | 191.50 ms | 199.94 ms | +8.44 ms |
| GroupedMatmul | 116.76 ms | 121.74 ms | +4.98 ms |

逐次 kernel 记录确认：

- 输入形状完全相同；
- 调用次数相同；
- Inductor 下平均和中位耗时仍慢约 4%；
- 其他 QuantBatchMatmul 没有普遍变慢，因此不是整卡统一降速。

**当前结论：**Norm/Slice 融合有效，但两类 MoE kernel 的回退抵消了约一半本可获得的 Computing 收益。

**尚未证实：**MoE 为什么变慢。下一步检查地址对齐、storage offset、数据布局，以及 default 防御性 Clone 与 Inductor 原地复用的差异。

## 7. 27B：长 prefill 的复制、激活和状态初始化回退

### 7.1 长 prefill 总体结果

| 指标 | default | Inductor | 变化 |
|---|---:|---:|---:|
| Stage | 28.24 s | 30.83 s | +2.59 s |
| Computing | 28.20 s | 30.79 s | +2.59 s |
| replay | 68 次 | 68 次 | 相同 |

replay 次数相同，因此本次回退不能归因于图捕获覆盖差异。

### 7.2 主要新增开销

| 新生成 kernel | Inductor 累计时间 | 相对 default 增量 |
|---|---:|---:|
| `unsafe_view_clone_view` | 1.374 s | +1.374 s |
| `mul...sigmoid_view` | 1.217 s | +1.217 s |
| `zeros_4` | 0.590 s | +0.590 s |
| `index_3` | 0.069 s | +0.069 s |

同时，一些原有算子确实减少：

- Slice：约 -0.175 s；
- TensorMove：约 -0.116 s；
- FusedInferAttentionScore：约 -0.164 s；
- QuantBatchMatmulV3：约 -0.085 s。

因此新增复制与初始化成本超过了融合和大算子小幅改善所节省的时间。

### 7.3 输入打包

`unsafe_view_clone_view` 将 `(token, head, channel)` 数据从大缓冲区重新打包，再交给 GDN LayerNorm。

生成代码中，输入按原始 token stride 16384 读取，输出按连续 token stride 6144 写入。现有二维 LayerNorm 的单一 `stride_z_row` 无法表达这种复合布局。

因此不能简单删除 `clone`。可行改造是让 LayerNorm 直接按以下方式读取：

```text
token × token_stride + head × head_stride + channel
```

快速检查还确认，三维输入直接 `reshape(-1,128)` 时已经发生连续化；不能把普通连续输入的 LayerNorm 测试当作该改造已经通过的证据。

### 7.4 Sigmoid+Mul

名字中虽然包含 `dynamic_quant`，但生成 kernel 实际只做：

1. 读取两个输入；
2. 对一个输入计算 Sigmoid；
3. 与另一个输入相乘；
4. 写回结果。

动态量化仍由后面的原生 `npu_dynamic_quant` 执行，不能把这段回退归因于量化本身。

下一步应比较长形状下原生 Sigmoid+Mul、当前 Triton kernel 和不同 tiling 配置，必要时按形状选择原生实现。

### 7.5 状态清零

`zeros_4` 创建的缓冲区会作为状态返回给后续执行，不能直接视为无效计算删除。

可以考虑：

- 使用原生清零；
- 优化状态初始化路径；
- 检查是否能与首次状态更新合并。

### 7.6 为什么 decode 略快但整体仍回退

固定 decode 窗口中：

- Stage 约下降 0.44%；
- Computing 基本持平；
- Free 略有下降。

正式非 Profile 数据也显示 median ITL 略快。但长 prefill 更慢，并增加后续长停顿，最终平均 TTFT、TPOT 和吞吐仍然回退。

## 8. VL：MRoPE 后的完整 QKV 复制是主要问题

### 8.1 长 prefill 总体结果

| 指标 | default | Inductor | 变化 |
|---|---:|---:|---:|
| Stage | 15.45 s | 17.63 s | +2.17 s |
| Computing | 15.41 s | 17.58 s | +2.17 s |
| replay | 33 次 | 33 次 | 相同 |

主要新增开销：

| 新 kernel | 时间增量 |
|---|---:|
| `copy_split_with_sizes_3` | +2.788 s |
| `copy_split_with_sizes_4` | +0.103 s |
| MRoPE 相关 Clone kernel | 约 +0.09 s |

与此同时：

- MatMulV3 约 -0.348 s；
- FusedInferAttentionScore 约 -0.157 s；
- TensorMove、Slice、ViewCopy 也有所减少。

VL 同样不是大计算算子回退，而是复制成本抵消了这些改善。

### 8.2 为什么完整 QKV 复制存在浪费

模型每层 QKV 共 4608 个通道：

- Q：前 3584 个通道；
- K：接下来 512 个通道；
- V：剩余 512 个通道。

MRoPE 只更新 Q 和 K，但当前生成 kernel 会：

1. 读取整个 4608 通道的原 QKV 缓冲区；
2. 用新的 Q/K 结果替换对应部分；
3. 保留不变的 V；
4. 把整个 4608 通道缓冲区重新写回。

生成代码还包含多次条件选择和对原缓冲区的读取。V 没有任何数值变化，却仍参与完整读取和写回。

优化顺序：

1. 优先验证 MRoPE 直接写回原 QKV 缓冲区；
2. 如果算子接口或别名规则不允许，只复制 Q/K；
3. 无论采用哪种方案，都跳过 V 的无效读写；
4. 修改后执行固定输入精度检查和至少三轮非 Profile 性能复测。

## 9. 下一步优化方向

| 优先级 | 优化方向 | 目标与依据 |
|---|---|---|
| P0 | VL：MRoPE 原地输出，或只复制 Q/K、跳过 V | 直接处理 Profile 中约 2.788 s 的主要复制开销 |
| P0 | 27B：长形状 Sigmoid+Mul 原生回退或专项 tiling | 处理约 1.217 s 的新增 kernel 开销 |
| P0 | 27B：优化 Zeros 状态初始化 | 处理约 0.590 s 的新增开销 |
| P1 | GDN LayerNorm 支持三维 token/head 索引 | 消除约 1.374 s 的输入打包 |
| P1 | 30B：检查 MoE 地址对齐、布局和 Clone 差异 | 收回约 13.4 ms 的 decode 回退 |
| P1 | 关键节点按形状选择原生实现或 Triton 融合 | 避免短形状加速、长形状回退 |

更通用的工程方向是：对少数已证明存在差异的节点，测量并缓存不同形状下的实现选择，而不是全局开启或关闭所有融合。

### 暂不建议优先投入

- **继续扩大长 prefill 图捕获**：最新责任点已经落在具体生成 kernel。
- **全局开启 Coordinate Descent**：此前没有明确性能收益，并会增加编译成本。
- **继续增加更多 Norm 融合**：30B 已有融合收益，当前更应解决 MoE 抵消和长形状复制。
- **仅根据算子数量判断优化效果**：少量低效大 kernel 可以抵消大量小算子消除。

## 10. 修改后的验证要求

每项优化至少执行：

1. **单元或小函数数值检查**：覆盖原始连续布局和新增非连续布局。
2. **固定输入精度**：使用相同 token ID，比较 token 与 logprob，不只看文本是否相似。
3. **相同形状 Profile**：确认目标 kernel 的时间确实下降，并检查是否把开销转移到其他节点。
4. **非 Profile 性能复测**：至少三轮，保持相同 QPS、并发和缓存状态。
5. **跨形状检查**：同时覆盖 ShareGPT 中短输入和 8K/32K 长输入，防止修复一个形状后损害另一个形状。
6. **结果解释限制**：Profile 时间只用于机制分析，正式收益仍以未开启 Profiler 的服务结果为准。

## 11. 结论边界

1. 最新每个场景执行了一对 default/Inductor Profile，用于定位主要责任点；具体 MoE 根因和优化后的收益仍需补充验证。
2. 各算子累计设备时间可能重叠，不能全部相加后当作端到端时间。
3. 早期 27B RMSNorm launcher 的巨大异常已经修复。当前 32K 回退是新的较小幅度问题，不能继续套用旧的 launcher 根因。
4. 早期“decode 单步完全不变”的结论不能机械套到当前所有模型。当前 30B 已出现稳定 median ITL 改善，需要以当前相同形状 Profile 为准。
5. Profile 中观察到的设备时间不能直接承诺为等量的端到端收益。
6. 最新分析没有修改模型或编译器源码。验证结束后，本任务使用的 NPU 0、1 均已释放。

## 12. 结果与脚本索引

### 本地结果

- 正式矩阵汇总：[20261126_perf_cap1024_SUMMARY.md](../tests/20261126_perf_cap1024_SUMMARY.md)
- 最新 Profile 与生成代码：`server-results/benchmark-diagnosis/20261126_inductor_optimization/`
- 30B：`qwen30b_decode/comparison/`
- 27B：`qwen27b_32k/comparison/`、`qwen27b_32k/generated_code/`
- VL：`qwen25vl_32k/comparison/`、`qwen25vl_32k/generated_code/`

主要文件：

- `comparison/summary.json`：Stage、Computing、Free、replay 和 launch 汇总；
- `comparison/operator_events.csv`：逐窗口、逐算子统计；
- `comparison/api_events.csv`：Host API 统计；
- `generated_code/**/output_code.py`：真实 Inductor 生成代码；
- `profile_summary.json`：请求 token 数、TTFT 和窗口完成情况。

### 临时分析脚本

`server-scripts/benchmark-diagnosis/20261126_inductor_optimization/`

- `profile_fixed_batch_cap1024.py`：固定 token ID、固定批次、双窗口 Profile；
- `compare_timed_profiles.py`：提取算子、API 和 Stage 汇总；
- `summarize_kernel_shapes.py`：按形状汇总逐次 kernel 时间。

### 相关历史分析

- [20261124_Inductor性能收益来源与捕获窗口对照.md](20261124_Inductor性能收益来源与捕获窗口对照.md)
- [20261123_Inductor三大优势场景收益来源.md](20261123_Inductor三大优势场景收益来源.md)
- [20261123_023141_Inductor新增优势场景证据.md](20261123_023141_Inductor新增优势场景证据.md)

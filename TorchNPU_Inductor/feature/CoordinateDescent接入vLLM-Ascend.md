# Coordinate Descent 接入 vLLM Ascend：实现、验证与价值评估

## 1. 文档状态

- 开发日期：2026-11-22。
- 服务器：A5-25，Ascend950DT，CANN 9.2.0。
- 容器：`inductor_feature_z00896543_20261122_140509`。
- 工作区：`/home/z00896543/validation/inductor_feature_20261122_140509/inductor-meta-worktree`。
- vLLM Ascend 开发分支：`hanya/feat/inductor-coordinate-descent`。
- 阶段提交：`7d17773a5339774fe1ac75412ea6f320decd4be5`。
- 对应 patch：`Patch/CoordinateDescent/0001-allow-coordinate-descent-tuning-on-npu.patch`。
- 当前结论：功能接入、单元测试、真实执行证据、小模型及 Qwen3-32B 对照均已完成；两种规模均未发现明确性能收益，因此保持默认关闭。

## 2. 背景：为什么补这个能力

Coordinate Descent 是 Inductor 在普通候选配置选择之后继续进行的局部搜索调优。它可以在已有候选结果附近继续调整 block 等参数，而不是只从第一轮候选列表中选一个配置。

上游 vLLM 对静态 `compile_sizes` 会通过 `set_inductor_config()` 注入：

- `max_autotune`；
- `coordinate_descent_tuning`。

对应开关为：

```bash
VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=1
VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=1
```

torch_npu 的 `triton_experimental` 后端已经实现这条路径。`NPUCachingAutotuner.run()` 在普通 autotune 选出候选后，检查 `inductor_meta["coordinate_descent_tuning"]`，调用 `coordinate_descent_tuning()`，再将返回结果包装成最终 launcher。调优结果也会通过 `found_by_coordesc` 写入 `.best_config`。

因此该工作不需要开发新的 NPU kernel 或新的调优算法，只需要恢复 vLLM Ascend 对已有能力的配置传递。

## 3. 原问题与代码原因

vLLM Ascend 的 `_setup_inductor_track_envs()` 原先已经会保留用户显式设置的：

```text
VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=1
```

但随后 `_normalize_inductor_config()` 又把 `coordinate_descent_tuning` 列在 `_TRACK_PINNED_OFF_INDUCTOR_KEYS` 中。只要对应值为真，就告警并改写为 `False`。

这意味着“环境变量被保留”并不等于“特性能够运行”：真正进入 Inductor 的 per-compile 配置仍被覆盖。

修改位置：

- `vllm_ascend/platform.py`：从强制关闭列表删除 `coordinate_descent_tuning`。
- `tests/ut/compilation/test_inductor_config_normalization.py`：验证 `True` 原样保留。
- `tests/ut/compilation/test_compile_backend_track.py`：验证上游显式环境开关原样保留。

本阶段只恢复显式 opt-in。默认值仍为 `False`，避免在收益未确认前增加所有用户的冷启动成本。

## 4. 环境初始化与版本固定

使用 meta 仓的环境 skill 初始化新容器，关键版本如下：

| 组件 | 版本或提交 |
|---|---|
| meta 分支 | `hanya/feature/upgrade_vllm_ascend_master_1120` |
| meta 提交 | `a71486223e28a6a4298946b6abe293a0350080bc` |
| pytorch 分支 | `hanya/fix/torch-npu-2.13.0-post1` |
| pytorch 提交 | `6f6a1bcc1217a51acf19ba188e0fdc898d37a7ea` |
| torch | `2.13.0+cpu` |
| torch_npu | `2.13.0.post1+git6f6a1bc`，从上述提交源码构建 |
| triton | `3.6.0` |
| vLLM | `0.33.0` |
| vLLM Ascend | `0.33.0`，editable 构建 |
| FLA NPU | `26.10.0+feature` |

构建日志和环境原始结果位于服务器的 `results/` 目录。安装依赖批次时检查了 torch、torch_npu 和 triton 的版本，避免依赖安装覆盖编译栈。

## 5. 基线 Qwen 小模型冒烟

模型：`/mnt/weight/Qwen/Qwen3-0.6B`。

运行配置：

```python
compilation_config={"backend": "inductor"}
```

基线结果：

- 两个中英文提示均生成非空文本；
- 日志出现 `ACL graph replay is active`；
- 编译缓存中有 10 个包含 `npu_triton_heuristics` 的 Python 生成产物；
- 因而确认不是仅设置了一个无效配置，而是确实进入 TE 生成代码和图回放路径。

同模型 eager 对照也能正常生成。eager 与 Inductor 的部分文本存在差异，但两者都有小模型常见的重复和偏题现象；本阶段没有把内容差异单独归因为 Coordinate Descent。后续开关对照使用完全相同的文本 hash 做更严格的检查。

修改前配置单元测试为 52 项全部通过；修改后为 54 项全部通过。

## 6. 直接证明 torch_npu 已有能力

先用独立的小函数验证 TE，而不经过 vLLM：

- 函数包含 `relu` 和 `sigmoid`；
- `torch.compile(..., options={"npu_backend": "triton_experimental", "coordinate_descent_tuning": True})`；
- 对 `NPUCachingAutotuner.coordinate_descent_tuning` 加调用计数；
- 编译结果与 eager 使用 `torch.testing.assert_close` 比较。

结果：

```text
COORDINATE_DESCENT_SMOKE_PASS calls=1
```

这一步证明后端实现本身可以执行，后面的 vLLM 测试才具有明确意义。

## 7. vLLM Ascend 中的真实生效证据

为了区分“配置存在”和“代码真正运行”，验证时临时在已安装的 TE `NPUCachingAutotuner.run()` 分支中加入日志探针。探针位于实际调用 `coordinate_descent_tuning()` 的代码分支内，不改变算法。该探针只用于测试，不属于阶段提交。

开启组：

```bash
VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=1
VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0
```

关闭组：

```bash
VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0
VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0
```

两组均使用：

- 独立 `VLLM_CACHE_ROOT`；
- `compile_sizes=[16]`；
- `cudagraph_capture_sizes=[16]`；
- 16 条固定提示；
- 每次最多生成 64 token；
- 开启和关闭各运行两轮。

结果：

| 证据 | 关闭组 | 开启组 |
|---|---:|---:|
| `TE_COORDINATE_DESCENT_EXECUTE` 日志数 | 0 | 58 |
| `found_by_coordesc=true` 的 `.best_config` 文件数 | 0 | 6 |
| 两轮输出在组内一致 | 是 | 是 |
| 两组最终输出 hash 一致 | 是 | 是 |

开启组输出 hash：

```text
0d6fda96802e5ddc4c238985346d5def3f83c650522f9b58823e1928cfe8e4cc
```

因此可以确认：配置不仅通过了 vLLM Ascend 的归一化，而且实际进入了 TE 的调优代码，并产生了由 coordinate descent 找到的配置。

## 8. 性能与启动收益

### 8.1 Qwen3-0.6B 小模型对照

| 指标 | 关闭组 | 开启组 | 开启/关闭 |
|---|---:|---:|---:|
| 启动时间 | 387.036 s | 430.553 s | 1.1124 |
| 两轮生成时间中位数 | 0.375856 s | 0.375565 s | 0.9992 |
| 输出 hash | 相同 | 相同 | — |

解释：

- 生成时间基本持平，差异不足以支持性能收益结论；
- 开启组启动时间增加约 11.2%；
- 小模型没有展示出值得默认开启的收益；
- 当前实现保持默认关闭是合理的。

原始数据归档：

```text
server-results/inductor-feature/20261122_140509/coordinate_small/
```

### 8.2 Qwen3-32B 大模型对照

测试模型：`/mnt/weight/Qwen3-32B`，BF16，单卡。

配置与第 7 节一致，但每组生成 128 token、重复三轮。

关闭组已经完成：

- 启动时间：3836.072 s；
- 三轮生成时间：3.233104 s、3.172925 s、3.175689 s；
- 中位数：3.175689 s；
- 三轮输出一致，输出内容正常；
- 输出 hash：`989e0fe71733fdca9ba94994050dc10613c20e6f22523ba73f8ef41f1472c8ae`。

开启组结果：

- 启动时间：3775.259 s；
- 三轮生成时间：3.240719 s、3.179980 s、3.182403 s；
- 中位数：3.182403 s；
- 三轮输出一致，输出内容与关闭组相同；
- 输出 hash 与关闭组完全相同；
- 日志中有 130 次 `TE_COORDINATE_DESCENT_EXECUTE`；
- 开启组有 6 个 `found_by_coordesc=true` 配置，关闭组为 0。

两组比较：

| 指标 | 关闭组 | 开启组 | 开启/关闭 |
|---|---:|---:|---:|
| 启动时间 | 3836.072 s | 3775.259 s | 0.9841 |
| 三轮生成时间中位数 | 3.175689 s | 3.182403 s | 1.0021 |
| 输出 hash | 相同 | 相同 | — |

解释：开启组生成时间约慢 0.21%，启动时间约快 1.59%。两个差异都很小，不能据此主张性能收益。特别是启动总耗时已超过一小时，单次运行的百分比差异不能替代重复启动分布。当前大模型结论仍是：功能真实生效，但没有明确性能收益证据。

原始数据归档：

```text
server-results/inductor-feature/20261122_140509/coordinate_qwen32b/
```

## 9. 当前价值判断与适用范围

已确认的价值：

- 用户现在可以通过上游现有开关使用 torch_npu 已有的 coordinate descent；
- 该功能在 vLLM 的静态尺寸编译路径上真实可用；
- 开关前后生成结果一致；
- 调优结果可以写入缓存并用于后续运行。

限制：

- 只对需要调优的 kernel 有潜在价值；如果普通候选已足够好，收益可能接近零；
- 静态 `compile_sizes` 会增加编译和首轮执行成本；
- 小模型上启动开销明确增加，生成收益接近零；
- Qwen3-32B 上生成和启动差异也都很小，没有明确性能收益；
- 大模型两组的启动时间都超过一小时，说明静态尺寸与自动调优成本本身很高；
- 不建议当前直接改成默认开启。

## 10. 测试命令

在服务器工作区中：

```bash
source env.sh
cd vllm-ascend
python -m pytest -q \
  tests/ut/compilation/test_inductor_config_normalization.py \
  tests/ut/compilation/test_compile_backend_track.py
```

本阶段结果：`54 passed`。

实际 vLLM 对照使用服务器脚本：

```text
/home/z00896543/validation/inductor_feature_20261122_140509/scripts/coordinate_bench.py
```

原始输出、缓存证据和阶段日志保存在同一部署根目录的 `results/` 下。

## 11. Patch 应用与回退

应用：

```bash
git apply Patch/CoordinateDescent/0001-allow-coordinate-descent-tuning-on-npu.patch
```

回退：

```bash
git revert 7d17773a5339774fe1ac75412ea6f320decd4be5
```

只关闭特性而不回退代码时，设置：

```bash
VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0
```

默认情况下不需要设置，因为本阶段保持默认关闭。

# AllenCGuan已有修改

## 历史修改分析

https://github.com/Allen-C-Guan/triton-ascend/tree/d12b7d9f200a8a7a9e106cb91c3b7c614ee99cbd 中顶端Guan Yingchen的几个提交主要在做什么？

顶端这 4 个提交（作者显示为 **Guan Yingchen / AllenGuanC**）主要不是开发新的编译器功能，而是在做 **NPU 扩展的编译环境适配**，以及 **AscendNPU-IR 子模块的来源和版本管理**。

按时间顺序看：

1. [`25f1ed0`：固定 AscendNPU-IR 版本](https://github.com/Allen-C-Guan/triton-ascend/commit/25f1ed0)
   将 `third_party/ascend/AscendNPU-IR` 子模块从 `f96183d` 更新到 `29d75b8`。属于依赖版本切换，本仓库中没有其他代码改动。
2. [`0d69f74`：解决 torch_npu 与 CANN 的 ACL 头文件不兼容](https://github.com/Allen-C-Guan/triton-ascend/commit/0d69f74)
   在 `third_party/ascend/backend/utils.py` 中新增 `_get_torch_npu_acl_inc_dirs()`：
   - 优先使用 `torch_npu` 自带的较新 ACL 头文件，避免旧版 CANN 缺少 `aclmdlRICondHandle` 等定义。
   - 通过 `/tmp` 下的软链接包装目录，只覆盖 `acl/` 头文件搜索路径。
   - 避免直接把整个目录加入 `-I`，否则其中的 `profiling/`  stub 头文件会遮蔽 CANN 的正式头文件。
   - 将该路径放在 CANN 头文件路径之前，用于修复 `npu_utils` 扩展编译。
3. [`5694ff6`：把 AscendNPU-IR 子模块切到个人 fork](https://github.com/Allen-C-Guan/triton-ascend/commit/5694ff6)
   修改 `.gitmodules`：
   - 来源从 `Ascend/AscendNPU-IR.git` 改为 `AllenGuanC/AscendNPU-IR.git`。
   - 明确跟踪 `master` 分支。
   - 保留浅克隆配置 `depth = 1`。
4. [`d12b7d9`：适配 torch_npu 新的 ACL 目录布局](https://github.com/Allen-C-Guan/triton-ascend/commit/d12b7d9f200a8a7a9e106cb91c3b7c614ee99cbd)
   这是对第 2 个提交的后续修正。新版 `torch_npu` 将完整 ACL 头文件放到了 `include/acl/`，旧路径只剩一个不完整子集，缺少 `aclrtStream`、`aclrtDrvMemHandle`、`aclprofStepInfo` 等定义。因此现在：
   - 优先选择 `include/acl/`；
   - 不存在时回退到旧的 `include/third_party/acl/inc/acl/`；
   - 继续使用软链接包装，避免影响其他头文件。

**概括来说：先更新并接管 `AscendNPU-IR` 依赖，再连续修复 `torch_npu` 新旧版本与 CANN 之间的 ACL 头文件搜索路径问题，目标是让后端 NPU 扩展能够正常编译。**

## 这几个提交在新版本的triton-ascend是不是用不到了

**对官方新版 `triton-ascend` 来说，基本用不到了，也不建议直接 cherry-pick。**我核对了最新正式版 `v3.6.0.post2` 和当前 `main`，情况如下：

- **`25f1ed0`：固定旧版 AscendNPU-IR**
  不需要。新版有自己的子模块版本，沿用旧的 `29d75b8` 反而可能造成依赖降级或不匹配。
- **`5694ff6`：切换到个人 fork**
  不需要。新版 [`.gitmodules`](https://raw.githubusercontent.com/triton-lang/triton-ascend/main/.gitmodules) 仍指向官方的 `Ascend/AscendNPU-IR.git`，没有使用个人 fork。
- **`0d69f74` + `d12b7d9`：ACL 头文件路径兼容补丁**
  新版已经不含 `_get_torch_npu_acl_inc_dirs()` 这套软链接包装逻辑。当前 `main` 的 [`utils.py`](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/backend/utils.py) 在构建 NPU 扩展时直接使用 CANN 的 `include`、`pkg_inc` 等路径，构建流程也已重构，旧补丁不能简单套用。

这两个 ACL 提交原本是针对 **torch_npu 2.13/master 与 CANN 9.1 beta 的特定头文件布局问题**；而当前官方 `main` 推荐的是 **CANN 9.2.0 + torch_npu 2.10.0.post7**，环境前提已经不同。

所以：

> **如果使用官方新版及官方匹配的 CANN、torch_npu，这四个提交都可以不带。**
> 只有在你自行搭配新版 `torch_npu master`，并再次出现 `aclrtStream`、`aclmdlRICondHandle` 等符号缺失时，才可能需要“类似思路”的适配，但也应基于新版构建代码重写，而不是直接合入旧提交。



# 3.6.0post2 的 AscendNPU-IR依赖

## bishengir-compile: Unknown command line argument '--enable-optimized-metaop=True'

直接使用3.6.0post2会出现编译错误：

bishengir-compile: Unknown command line argument '--enable-optimized-metaop=True'

Triton 向 `bishengir-compile` 传入 `--enable-optimized-metaop=True`，但该 pin 构建出的二进制只提供 `--enable-optimize-math` 等相关选项，没有 `optimized-metaop`。

Triton Ascend 采用“CANN ≥ 9.2 就启用 optimized-metaop”的假设，但当前 pin 的 `bishengir-compile` 与该假设不一致，导致 Triton 生成的命令行不能被 AscendNPU-IR 接受。

`optimized-metaop` 是做什么的？它用于选择 **`meta_op` 模板 bitcode 的优化版本**：

- `--enable-optimized-metaop=True`：选择 **O2 优化版**
- `--enable-optimized-metaop=False`：选择 **O0 非优化版**

它不是“开启或关闭 metaop”，也不是 Qwen/Inductor 的功能开关，而是决定 `bishengir-compile` 在 A3/A5 编译时向模块链接哪一种 meta_op 模板实现。

AscendNPU-IR 的支持提交 `9607e1add5973da598f5b5c28f546d2fd4cac424` 明确说明了这一点：

> 用布尔参数 `--enable-optimized-metaop` 替代原来的 O2/O0 字符串参数；true 对应 O2，false 对应 O0。

后续提交 `65fdf2debbf81c934da95c0c147c2c1415ad4395` 又加入了 CANN 版本自动检测：

- CANN ≥ 9.2：支持并使用 O2 meta_op bitcode
- 更老 CANN：使用 O0 版本

为什么对不上？当前triton-ascend 的节点3.6.0post2虽然是发布节点，但是他发布时并没有同步更新submodule AscendNPU-IR的节点pin，导致如果从submodule构建 AscendNPU-IR就会出现版本不兼容

证据链是：

- **10 月 29 日 16:10**：AscendNPU-IR 的 `4eb12142` 已经存在。
- **10 月 29 日 17:12**：Triton 发布 `v3.6.0.post2`，此时 submodule 仍指向旧的 `bb13ef612052`。
- **10 月 29 日 21:42**：有人提交了“把 submodule 更新到 `4eb12142`”的修改。
- **10 月 31 日**：这个修改合入 Triton `main`。



##  `triton-mlir-opt` 时缺少 `mlir::hivmave::AVEDialect` 符号

- **2026-10-26**，`upstream/main` 通过 `97ec8d2d`（PR `#2746`）删除了 `triton-mlir-opt` 及其 CMake 目标。
- **2026-10-29**，通过 `68a9023c`（PR `#2826`）给 `TritonAscend` 插件补上 `MLIRAVEDialect`；该提交进入仓库的 committer 时间是 **2026-10-31**。

所以严格来说，**上游没有单独修复 release 中的 `triton-mlir-opt` 链接问题**：上游先删除了这个工具，再为保留的 `TritonAscend` 插件补齐 AVE 支持。release/3.6.0 保留了工具，却没有同步这些调整，才在升级 IR 后暴露问题。

对于当前 release，最合适的是**不 cherry-pick 其中任何一个**，而是新增一个最小提交，只修改：

```
third_party/ascend/bin/CMakeLists.txt
```

在 `triton-mlir-opt` 的 `target_link_libraries` 中加入：

```
MLIRAVEDialect
```

总结：**`97ec8d2d` 是删除工具的替代路线；`68a9023c` 修的是插件；当前问题需要单独补 `triton-mlir-opt` 的链接依赖。**
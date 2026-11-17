# inductor是什么？为什么要引入inductor？

**Inductor 是 PyTorch 2.x 引入的默认深度学习编译器后端**，也是 `torch.compile()` 通常使用的核心编译组件。

它做的事情可以简单理解为：

> 把 PyTorch 模型中的计算图，转换成经过优化的高性能机器代码，而不是逐个执行原始 PyTorch 算子。

1. **TorchDynamo 捕获计算图**：在尽量保持 Python 使用方式不变的情况下，提取模型中的 PyTorch 计算。
2. **Inductor 分析和优化计算图**：进行算子融合、内存访问优化、循环调度等。
3. **生成目标代码**：
   - GPU 上通常生成 **Triton** 内核；
   - CPU 上通常生成 **C++/OpenMP** 代码。
4. **编译并执行**：后续运行使用编译后的代码，而不是完全按原来的逐算子方式执行。



# 目前vllm-ascend的编译路径

**Inductor 是一种图编译后端，但图编译不等于必须使用 Inductor。**

Ascend 会设置：

```
compilation_config.oot_compiler = "vllm_ascend.compilation.compiler_interface.AscendCompiler"
compilation_config.use_inductor = False
```

也就是用 **AscendCompiler 替换 Inductor 后端**。官方文档也明确说明了这一点：[Graph Mode Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)。

## 当前的图优化链路

可以概括为：

```
PyTorch 模型
    │
    │ vLLM / Dynamo 图捕获、FX 图划分
    ▼
AscendCompiler
    │
    ├── Npugraph_ex：较完整的 FX 图优化
    │       或
    └── GraphFusionPassManager：指定的 FX 融合 Pass
    │
    ▼
优化后的计算图 / Ascend 自定义算子
    │
    │ ACLGraph（torch.npu.NPUGraph）捕获
    ▼
运行时 Replay
```



这里要区分两个阶段：

- **编译期图优化**：Npugraph_ex 或 FX fusion passes，负责算子融合、图改写等。
- **运行时图优化**：ACLGraph 负责捕获和回放，主要减少 Host 侧算子下发和 launch 开销，**它本身不是类似 Inductor 的 kernel 编译器**。

AscendCompiler和Inductor的核心区别：**AscendCompiler 主要是“改写 FX 计算图，再调用已有 NPU 算子”；Inductor 则会进一步把计算 lowering、调度，并生成新的 kernel 代码。**

# **Inductor 的关键概念**

### 1. FX Graph、ATen Operator 与 Decomposition

这是理解 Inductor 输入的基础。

PyTorch 的 FX Graph 通常由节点组成，每个节点代表一个操作。

`aten.add.Tensor`、`aten.relu.default` 是 ATen Operator。

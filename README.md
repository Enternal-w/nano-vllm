<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## 重点关注
A. 内存管理模块 (BlockManager)
这是灵魂所在。关注它是如何计算 GPU 还能容纳多少个 Block 的。

重点关注：如何维护 free_blocks 列表，以及在 forward 之前如何根据输入长度分配 block_table。

B. 算子实现层 (nanovllm/layers/attention.py)
这是最硬核的部分。通常会调用自定义的 CUDA Kernel（或者 Triton 实现）。

重点关注：Attention 类是如何接收 block_table 的。你会发现它不再是简单的 torch.nn.MultiheadAttention，而是需要处理不连续内存的特殊逻辑。

C. 调度逻辑 (Scheduler)
决定哪些请求（Prompts）可以进入当前的 Batch。

重点关注：当显存不足以容纳下一个生成的 Token 时，它是如何处理的（比如抢占 Preemption：暂停某些任务并释放其 Block）。

D. 模型并行与定义 (nanovllm/layers/linear.py)
代码中，ColumnParallelLinear 和 RowParallelLinear 是分布式推理的基础。

重点关注：All-Reduce 操作发生在哪里。

这是为您准备的 GitHub README 技术深度解析文档。它基于您对 `nanovllm` 源码的深入研究，采用 Infra 后端开发视角，系统地总结了推理引擎的核心机制。



## 🚀 NanoVLLM 技术核心深度解析笔记

本仓库是对高性能 LLM 推理引擎（如 vLLM）核心原理的轻量化实现与深度剖析。通过阅读源码，我针对显存管理、请求调度及分布式并行等方面总结了以下核心技术要点：

### 1. 显存管理：基于 PagedAttention 的虚拟化解耦

在传统推理框架中，KV Cache 的连续存储会导致严重的内部与外部显存碎片。`nanovllm` 通过 `BlockManager` 实现了物理显存与逻辑序列的彻底解耦：

* **分页存储机制**：将显存划分为固定大小的物理块（Blocks），通过 `free_blocks` 队列进行动态管理。系统不再预留固定最大长度的连续空间，而是按需分配物理块，将显存利用率提升至接近 100%。
* **逻辑映射与 BlockTable**：每个序列维护一个 `block_table`，记录了其逻辑顺序对应的物理块索引。这种映射允许算子（如 FlashAttention）在非连续的物理地址上执行高效的注意力计算。
* **Prefix Caching（前缀缓存）**：利用 `compute_hash` 对 Token 序列进行哈希识别。在多轮对话中，如果不同请求共享相同的 System Prompt 或历史上下文，系统通过增加 `ref_count` 直接复用已存在的物理块，不仅节省了宝贵的显存，还消除了冗余的计算开销。

### 2. 调度逻辑：Continuous Batching 与资源水位策略

`Scheduler` 是整个推理引擎的控制大脑，其核心在于平衡系统吞吐量与推理延迟：

* **动态 Batching 引擎**：不同于传统的静态 Batch，`Scheduler` 实时监控 GPU 资源水位（即 `BlockManager` 中的剩余可用块数）。只要资源满足 `can_allocate` 条件，新请求即可进入 Pre-fill 阶段。
* **Prefill 与 Decode 优先级仲裁**：系统优先处理处于 Pre-fill 阶段的请求，以充分压榨 GPU 的计算密集型算力；随后进入 Decode 阶段进行逐字生成，此时任务转变为访存密集型。
* **抢占机制 (Preemption)**：当显存水位达到极限，无法支撑现有请求继续生成下一个 Token 时，`Scheduler` 会执行抢占逻辑。它会牺牲最晚进入队列的请求，通过 `deallocate` 释放其占用的 Block 并将其状态重置为 `WAITING`，确保系统在高并发极端情况下不会因 OOM（显存溢出）而崩溃。

### 3. 分布式并行：张量并行 (TP) 与通信掩盖

为了支持单卡显存无法容纳的大模型，项目实现了基于 Megatron-LM 风格的张量并行策略：

* **行列分片模式**：通过 `ColumnParallelLinear` 在输出维度切分权重，以及 `RowParallelLinear` 在输入维度切分权重。这种设计保证了在两次矩阵乘法之间无需跨卡通信。
* **通信瓶颈优化**：识别出 `All-Reduce` 操作是 TP 并行的主要耗时点。在 `RowParallelLinear` 的 `forward` 结尾触发同步，将各卡的局部结果进行汇总。深入理解了跨卡带宽（如 NVLink）对并行加速比的关键影响。

### 4. 高性能算子：Triton Kernel 与 Paged 存取

* **高效写入 (Scatter-Store)**：利用 **Triton** 编写了自定义的 `store_kvcache_kernel`。该算子根据 `slot_mapping` 将计算出的新 K/V 向量以“散射”方式精准存入分散的物理槽位，有效规避了原生 PyTorch 索引赋值带来的高昂 Kernel Launch 开销。
* **变长注意力适配**：集成 **FlashAttention-2** 的变长序列接口（Varlen），通过 `block_table` 引导算子在非连续内存布局上完成高效计算，确保了分页管理模式下的推理性能依旧接近硬件峰值。


## 学习总结

* BlockManager实现了逻辑内存与物理显存的解耦，把显存切成Blocks，按需分配。它还引入了 Prefix Caching（前缀缓存）。如果两个请求的开头是一样的（比如相同的系统提示词），它们会共享同一物理块的引用，不仅省显存，还省去了重复计算。

* Scheduler基于资源的动态 Batching 引擎，负责 Prefill 和 Decode 的优先级仲裁。
动态性：它不是固定 Batch Size，而是实时计算 GPU 剩下的房间（Blocks）够不够。
抢占机制 (Preemption)：如果显存满了，它会牺牲“最晚进来”的

* Tensor Parallelism通过 Megatron-LM 风格的行列分片，解决了单卡放不下大模型权重的问题。ColumnParallel：把矩阵纵向切开，大家算完后各拿一部分结果（无需通信）。
RowParallel：把矩阵横向切开，大家算出局部和，最后通过 All-Reduce 汇总。TP 的性能瓶颈往往不在计算，而在 All-Reduce 的跨卡带宽。

* Attention & Triton  这是 PagedAttention 的底层落地，利用 Triton 解决非连续显存的读写效率。写入 (Triton)：当模型算出一个新词时，它不是连续存的。Triton Kernel 负责根据 slot_mapping 像“散弹枪”一样精准地把 KV 存进物理块。
读取 (FlashAttention)：在计算注意力时，它根据 block_table 去不同的房间找数据。FlashAttention 算子保证了即使内存不连续，计算依然能维持极高的硬件利用率。

传统的模型推理受限于 KV Cache 对显存的静态预分配，导致严重的碎片化，限制了并发。所以引入了分页管理，将显存切块。配合动态调度器，根据剩余块数决定 Batch Size，并在极限情况下通过抢占确保系统不宕机。底层的计算不能再用原生的 PyTorch，必须使用 PagedAttention 思想的算子，利用 Triton 编写 Kernel 来处理这种非连续的 KV Cache 读写。当模型单卡放不下时，再通过张量并行（Column/Row Parallel）将负载平摊到多卡。

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
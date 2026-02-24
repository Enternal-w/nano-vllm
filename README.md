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
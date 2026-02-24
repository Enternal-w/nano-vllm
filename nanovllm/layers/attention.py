import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

'''
A. 为什么需要 Triton 实现的 store_kvcache？
传统的 PyTorch 操作（如索引赋值 cache[slot] = data）在处理数万个不连续的 slot 时，会产生大量的 GPU kernel launch 开销。
重点：Triton 允许我们将所有的存入操作合并为一个并行度极高的算子（每个 token 一个线程块），极大地减少了写缓存的延迟。

B. Prefill 和 Decode 的差异化算子
Prefill (flash_attn_varlen_func)：此时 Q 的序列长度和 K 一样长（都是 Prompt 长度）。它主要利用计算密集型优势，快速计算注意力。
Decode (flash_attn_with_kvcache)：此时 Q 只有一个 token，但 K 是过去所有 token 的集合（Paged Cache）。
重点：block_table 在这里被传入算子，这意味着 FlashAttention 本身必须支持非连续的内存访问（即通过索引表跳转寻找 KV 数据）。

C. Slot Mapping 与 Block Table 的配合
这是之前看的 BlockManager 的落地：
slot_mapping 告诉 Triton 每一个 token 具体写到显存的哪一个偏移位置。
block_tables 告诉 FlashAttention 每一组 token（Block） 在显存的什么位置。
核心联系：一个是 Token 级的写入（Write），一个是 Block 级的读取（Read）。
'''

@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,  # 当前步计算出的 K 的指针和步长
    value_ptr, value_stride,  # 当前步计算出的 V 的指针和步长
    k_cache_ptr, v_cache_ptr,  # 预分配的 KV Cache 大池子的指针
    slot_mapping_ptr,  # 映射表：逻辑位置 -> 显存物理位置
    D: tl.constexpr,  # 向量总维度 (heads * head_dim)
):
    # 1. 获取当前处理的 token 索引
    idx = tl.program_id(0)
    # 2. 从 slot_mapping 中加载该 token 对应的物理显存槽位地址
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return  # 忽略填充或无效位

    # 3. 计算当前 K, V 向量在内存中的偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 4. 加载计算出的局部 K, V 值
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 5. 计算并存储到 KV Cache 大池子的物理偏移位置（Slot 决定了去哪个块）
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale  # 缩放系数 (1/sqrt(dk))
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])   # 初始化缓存占位

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 1. 获取全局上下文（包含当前 batch 的各种索引信息）
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 2. 如果已分配缓存池，将本次计算的 k, v 存入池中
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 3. 根据阶段选择不同的 FlashAttention 优化算子
        if context.is_prefill:
            # --- Prefill 阶段 (处理 Prompt) ---
            if context.block_tables is not None:    # prefix cache  # 如果有前缀缓存命中
                k, v = k_cache, v_cache          # 直接从缓存取 KV，无需重复计算

            # 使用变长 FlashAttention，能够一次性处理 batch 中长度不一的 prompt
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)  # 传入块表支持分页
        else:    # decode
            # --- Decode 阶段 (逐字生成) ---
            # 使用专为解码优化的 FlashAttention，支持 Paged KV Cache
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens,   # 当前每个序列的有效长度
                                        block_table=context.block_tables,     # 逻辑块到物理块的映射
                                        softmax_scale=self.scale, causal=True)
        return o

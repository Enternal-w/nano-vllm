import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384  # 一批推理任务中允许的最大总 Token 数（影响吞吐量与显存占用）
    max_num_seqs: int = 512  # 一次推理批次（Batch）中允许的最大序列（句子）数量
    max_model_len: int = 4096  # 模型处理的最大上下文长度（长序列会被截断至此长度）
    gpu_memory_utilization: float = 0.9  # GPU 显存利用率（0.9 表示推理引擎将预分配并接管 90% 的可用显存）
    tensor_parallel_size: int = 1  # 张量并行度（模型权重被切分到多少张 GPU 上）
    enforce_eager: bool = False  # 是否强制使用 Eager 模式（True 则禁用 CUDA Graph 优化，方便调试但速度慢）
    hf_config: AutoConfig | None = None  # 存储从 HuggingFace 配置文件中加载的模型元数据对象
    eos: int = -1  # 终止符 ID（End of Sentence），默认为 -1
    kvcache_block_size: int = 256  # KV Cache 分块管理中每个 Block 的 Token 数量（用于 PagedAttention）
    num_kvcache_blocks: int = -1  # 显存中预分配的 KV Cache Block 总数（-1 表示根据剩余显存动态计算）

    def __post_init__(self):
        # 1. 校验模型路径必须是一个存在的目录
        assert os.path.isdir(self.model)
        # 2. 确保 KV Cache 块大小是 256 的倍数（为了对齐 GPU 算子性能）
        assert self.kvcache_block_size % 256 == 0
        # 3. 限制张量并行规模在 1-8 之间（通常对应单机最大 GPU 数）
        assert 1 <= self.tensor_parallel_size <= 8

        # 4. 从本地路径加载 HuggingFace 模型配置
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 5. 动态调整最大长度：取用户设定值与模型原生支持长度的最小值
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 6. 安全检查：单批次最大 Token 量不能小于单条序列的最大长度
        assert self.max_num_batched_tokens >= self.max_model_len

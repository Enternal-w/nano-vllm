import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

# 从自定义库 nanovllm 导入优化的算子
from nanovllm.layers.activation import SiluAndMul  # 融合的 SiLU 和 乘法算子（SwiGLU）
from nanovllm.layers.attention import Attention    # 核心注意力机制算子
from nanovllm.layers.layernorm import RMSNorm      # 均方根归一化
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear # TP 线性层
from nanovllm.layers.rotary_embedding import get_rope # 旋转位置编码
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead # TP 词嵌入与输出头


class Qwen3Attention(nn.Module):
    def __init__(
            self,
            hidden_size: int,  # 隐藏层维度
            num_heads: int,  # 总查询（Q）头数
            num_kv_heads: int,  # 总键值（KV）头数
            max_position: int = 4096 * 32,  # 最大序列长度
            head_dim: int | None = None,  # 每个头的维度
            rms_norm_eps: float = 1e-06,  # RMSNorm 的 epsilon
            qkv_bias: bool = False,  # 是否使用偏置
            rope_theta: float = 10000,  # RoPE 的基数
            rope_scaling: tuple | None = None,  # RoPE 缩放配置
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()  # 获取张量并行的 GPU 数量
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 当前进程分配的 Q 头数

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 当前进程分配的 KV 头数

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5  # 注意力评分缩放系数 (1/sqrt(d_k))
        self.qkv_bias = qkv_bias

        # 并行 QKV 投影层（列并行）
        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias,
        )
        # 输出投影层（行并行），之后接 All-Reduce
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
        )
        # 获取旋转位置编码实例
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim, max_position=max_position,
            base=rope_theta, rope_scaling=rope_scaling,
        )
        # 注意力计算核心
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

        # 如果没有 bias，Qwen 通常在 Q/K 上应用额外的 RMSNorm (QK Norm)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. 投影到 QKV 空间
        qkv = self.qkv_proj(hidden_states)
        # 2. 按照 Q, K, V 的维度进行切分
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. Reshape 为 [token_count, head_num, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 4. QK Norm（如果需要）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 5. 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 6. 注意力计算
        o = self.attn(q, k, v)
        # 7. 合并多头并投影回隐藏层维度
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        # 门控和上升投影层合并为一个并行线性层（SwiGLU 结构）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
        )
        # 下降投影层（行并行）
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul() # 执行 SiLU(gate) * up 的操作

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up) # 激活并相乘
        x = self.down_proj(x)    # 投影回原维度
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 初始化自注意力模块
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # 初始化 MLP 模块
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 层前归一化 (Pre-LN) 和 注意力后归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 这里的实现支持残差传递，以优化显存或加速
        # 1. 输入归一化 + 保存残差
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 2. 执行注意力
        hidden_states = self.self_attn(positions, hidden_states)
        # 3. 注意力后归一化并累加残差
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 4. 执行 MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 并行词嵌入
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 堆叠解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层归一化处理
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # 定义权重加载时的映射关系，用于自动合并 HuggingFace 格式的权重
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # 并行语言模型头（计算 logits）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 如果配置要求权重共享（词嵌入与输出头一致）
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 获取模型最后一层的隐藏状态
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 将隐藏状态映射到词表大小，获取分数
        return self.lm_head(hidden_states)

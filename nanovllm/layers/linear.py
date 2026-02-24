import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

'''
A. 为什么需要 Column 和 Row 并行成对出现？
Transformer 的每一层通常是：
1.ColumnParallel (如 QKV 投影)：将隐藏维度扩展，不需要同步。
2.RowParallel (如 O 投影)：将维度还原，并在结尾执行 all_reduce。
重点：这种设计使得在两次矩阵乘法之间不需要额外的通信。只有在 RowParallel 结束时才进行一次大通信。

B. All-Reduce 的性能开销
在 RowParallelLinear 的 forward 中，dist.all_reduce(y) 是唯一的跨卡通信点。

关注点：这是推理延迟（Latency）的主要来源。在多机多卡环境下，这里的带宽瓶颈直接决定了生成速度。这也是为什么 nanovllm 可能会使用更高效的自定义通信算子。

C. 权重加载逻辑 (weight_loader)
在普通的 PyTorch 代码中，加载权重就是 load_state_dict。但在 TP 环境下：
挑战：HuggingFace 的权重文件通常是单卡完整的。
解决方案：代码通过 narrow 和 chunk 手动将完整矩阵切碎分配给不同的进程。这也就是为什么在 LinearBase 中要把加载逻辑绑定到 Parameter 上——为了实现“按需切分加载”。
'''


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim  # 权重分片的维度（0是行，1是列，对应 output/input）
        self.tp_rank = dist.get_rank()  # 当前 GPU 的 ID
        self.tp_size = dist.get_world_size()  # 并行总 GPU 数
        # 初始化一个空的 Parameter，后续通过 loader 填充
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """不分片的线性层，每张显卡上都有一份完整的权重"""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)  # 直接拷贝

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 每个 GPU 只负责输出维度的一部分（output_size // tp_size）
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # 当前分片的大小
        start_idx = self.tp_rank * shard_size  # 根据 rank 计算切片起始位置
        # 从完整权重中切出属于当前 GPU 的部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 矩阵乘法：[batch, input] * [input, output/tp_size]^T
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """用于合并层，如 MLP 中的 Gate 和 Up 投影"""
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 计算该 shard 在合并权重中的偏移量
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 定位到本进程参数矩阵的子区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将完整权重切块并取出对应 rank 的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """专门为 Attention 的 Q/K/V 设计的并行层"""
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 逻辑与上面类似，但处理的是 "q", "k", "v" 字符串标识的切片
        # 确保 Q、K、V 每个部分都按 tp_size 被正确切分
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度被切分：每个 GPU 只接收输入向量的一部分
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 GPU 计算局部结果
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 核心,  同步所有 GPU 上的结果并相加
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

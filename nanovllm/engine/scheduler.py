from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


'''
在 Scheduler 中，需要重点关注 vLLM 是如何实现高并发与稳定性平衡的：

A. 抢占机制（Preemption）是如何运作的？
这是回答“显存不够了怎么办”的核心：

触发条件：在 Decode 阶段，如果显存池（BlockManager）无法为任何一个请求分配下一个 Token 的存储空间（can_append 失败）。

牺牲策略：代码选择了踢掉 running 队列中最后的序列。这通常是“最晚进来”的请求。

代价：被抢占的请求会丢失已计算的所有 KV Cache。当下一次重新调度它时，需要重新从头开始计算（Re-computation）。

B. Pre-fill 与 Decode 的优先级分配
代码中 if scheduled_seqs: return scheduled_seqs, True 这一行非常关键。它表明 Pre-fill（首词计算）的优先级高于 Decode（持续生成）。

逻辑：一旦有新任务且资源允许，调度器会优先把新任务塞进 Batch。这有助于提高吞吐量，但可能会导致正在生成的旧任务因为显存竞争被抢占。

C. 动态 Batch 的规模控制
max_num_batched_tokens 限制了计算负载。

max_num_seqs 限制了并发规模。

调度器在每一轮推理迭代前都会重新评估资源，这使得 Batch Size 是动态变化的，而不是像传统框架那样固定死。


3. 总结
Scheduler 通过 BlockManager 实时监控显存。它像一个精明的管家：

有新客（Waiting）且有余房（Blocks），立刻安排入住（Allocate）。

老客（Running）要续房（Append Token）但没空房了，就委屈后来的老客先退房（Preempt）。

客人退房（Finished），立刻打扫房间归还空房（Deallocate）。
'''


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 限制 Batch 中最大的请求数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 限制一次 Batch 处理的总 Token 数
        self.eos = config.eos # 终止符 ID
        # 初始化显存管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 待处理队列：存放新来的或被抢占的请求
        self.waiting: deque[Sequence] = deque()
        # 运行中队列：正在进行生成（Decode）的请求
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
                核心逻辑：优先处理等待中的请求（Pre-fill），如果没得处理，再处理运行中的请求（Decode）。
                返回: (选中的序列列表, 是否为 Pre-fill 阶段)
                """
        # prefill
        # --- 阶段 1: Pre-fill (首词生成) ---
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查资源：总 Token 数是否超标，显存块是否足够分配
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)    # 正式分配物理显存块
            # 计算实际需要计算的 Token (扣除掉命中缓存的前缀)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:  # 如果有新请求进来，优先做 Pre-fill（为了高吞吐）
            return scheduled_seqs, True

        # decode
        # --- 阶段 2: Decode (后续 Token 生成) ---
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查显存是否能容纳下一个生成的 Token
            while not self.block_manager.can_append(seq):
                # 【关键】抢占机制：如果显存不够新 Token 放入，就牺牲当前最晚进入队列的请求
                if self.running:
                    self.preempt(self.running.pop())    # 踢掉 running 队列末尾的
                else:
                    self.preempt(seq)      # 连自己都保不住，踢掉自己
                    break
            else:
                # 显存足够，允许该请求生成下一个词
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))       # 将选中的请求放回队列首部
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占逻辑：强制让出显存块，将请求打回等待队列"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)  # 释放 KV Cache 显存块
        self.waiting.appendleft(seq)  # 放在等待队列首部，保证下次优先调度

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """每一步推理完成后，更新状态"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 将生成的词加入序列
            # 检查是否结束（遇到 EOS 或达到最大长度）
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 彻底释放显存
                self.running.remove(seq)

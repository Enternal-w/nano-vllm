from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

'''
A. 逻辑地址与物理地址的解耦 (The Mapping)
注意 seq.block_table.append(block_id)。
对于模型层来说，它看到的 KV Cache 是连续的。
但在这里，block_table 存储的是一串物理 ID（如 [7, 102, 45]）。
意义：这实现了真正的分页管理。当显存不足时，管理器可以只给序列分配一个极小的块（256 tokens），而不需要预留 4096 tokens 的大空间。

B. 前缀缓存复用 (Prefix Caching)
注意 compute_hash 和 hash_to_block_id。
如果你输入两个 Prompt，开头都是“请帮我写一段代码...”，这段代码会识别出第一个 Block 的哈希值是一样的。
结果：allocate 方法会直接增加该块的 ref_count，而不会重新分配空间。模型推理时，直接跳过这些 Token 的计算（Pre-fill 阶段提速）。

C. 引用计数回收机制 (Ref Counting)
注意 deallocate 方法中的 block.ref_count -= 1。
只有当所有引用这个前缀的序列都结束了，这个物理块才会回到 free_block_ids。
意义：这保证了多用户并行请求时，共享资源的安全释放
'''



class Block:

    def __init__(self, block_id):
        self.block_id = block_id  # 物理块索引
        self.ref_count = 0  # 引用计数（为 0 时表示可回收）
        self.hash = -1  # 当前块内容的哈希值，用于缓存匹配
        self.token_ids = []  # 存储该块包含的具体 Token ID

    def update(self, hash: int, token_ids: list[int]):
        """更新块内容和哈希"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 预分配所有物理块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希映射表：用于快速找到内容相同的缓存块
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块队列（双端队列，方便取用和归还）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用的块集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 Token 序列的哈希值。
        prefix 是前一个 Block 的哈希，确保了链式依赖（即前缀必须完全一致）。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为新序列分配物理块表"""
        assert not seq.block_table
        h = -1
        cache_miss = False  # 标记是否发生缓存失效
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有填满的块才参与全局哈希缓存
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # 如果没命中缓存或内容不符
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 缓存失效，从空闲池取一个新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中！复用该块，增加引用计数，减少计算量
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:  # 更新哈希映射
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """推理过程中动态增加 Token 时，判断是否需要新开一个 Block"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        # 情况1：当前块刚满，新生成的 Token 需要开辟新物理块
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        # 情况2：当前块刚好填满，计算其哈希值并存入缓存表，供未来复用
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

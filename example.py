import os
from nanovllm import LLM, SamplingParams  # 导入核心推理引擎和采样参数类
from transformers import AutoTokenizer     # 导入 HuggingFace 的分词器工具


def main():
    # 设定模型权重存放的本地路径
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # 加载分词器，用于处理对话模板和文本转 ID
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 初始化 LLM 推理引擎
    # path: 模型路径
    # enforce_eager=True: 强制使用 eager 模式（不使用 CUDA Graph 优化，方便调试）
    # tensor_parallel_size=1: 张量并行度为 1（仅使用单张 GPU）
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 设置采样参数
    # temperature=0.6: 控制生成随机性（越低越严谨，越高越有创意）
    # max_tokens=256: 限制单次生成最大长度为 256 个 token
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 定义原始提示词列表
    prompts = [
        "introduce yourself",  # 自我介绍
        "list all prime numbers within 100",  # 列出 100 以内的质数
    ]

    # 应用聊天模板 (Chat Template)
    # 将原始问题包装成模型能理解的格式，例如：<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # 构造对话角色
            tokenize=False,  # 返回字符串而非 token 序列
            add_generation_prompt=True,  # 添加暗示模型开始回答的引导符
        )
        for prompt in prompts
    ]

    # 执行推理生成
    # llm.generate 会处理批处理逻辑，返回生成结果对象列表
    outputs = llm.generate(prompts, sampling_params)

    # 遍历并打印结果
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")  # 打印格式化后的提示词
        print(f"Completion: {output['text']!r}")  # 打印模型生成的文本内容


if __name__ == "__main__":
    main()

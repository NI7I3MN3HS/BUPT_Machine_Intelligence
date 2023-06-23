from utils import get_content_between_a_b, get_api_response
import torch

import random

from sentence_transformers import util


class RecurrentGPT:
    def __init__(self, input, short_memory, long_memory, memory_index, embedder):
        """
        初始化函数，用于创建RecurrentGPT类的实例
        参数：
        - input: 输入对象，包含各种输入信息
        - short_memory: 短期记忆，存储当前记忆的简要总结
        - long_memory: 长期记忆，存储之前的记忆内容
        - memory_index: 记忆索引，用于检索相似的记忆段落
        - embedder: 文本嵌入器，用于将文本转换为向量表示
        """
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.embedder = embedder
        # 如果存在长期记忆并且没有指定记忆索引，则使用embedder对长期记忆进行编码
        if self.long_memory and not memory_index:
            self.memory_index = self.embedder.encode(
                self.long_memory, convert_to_tensor=True
            )
        # 输出字典，用于存储分析后的输出结果
        self.output = {}

    # 准备输入，用于生成下一轮输入
    def prepare_input(self, new_character_prob=0.1, top_k=2):
        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]

        instruction_embedding = self.embedder.encode(
            input_instruction, convert_to_tensor=True
        )

        # 从记忆中获取前 3 个最相似的段落

        memory_scores = util.cos_sim(instruction_embedding, self.memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx]

        # 合并前 3 段

        input_long_term_memory = "\n".join(
            [
                f"Related Paragraphs {i+1} :" + selected_memory
                for i, selected_memory in enumerate(top_k_memory)
            ]
        )
        # 随机决定是否应该引入一个新角色
        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the output paragrah and add it into the memory."
        else:
            new_character_prompt = ""

        input_text = f"""我需要你帮我写小说。
        现在我给你一个400字的记忆（一个简短的总结），你应该用它来存储所写内容的关键内容，以便你可以跟踪很长的上下文。 
        每一次，我都会给你你当前的记忆（对以前故事的简要总结）。
        你应该用它来存储所写内容的关键内容，这样你就可以跟踪很长的上下文），之前写的段落，以及下一段要写什么的说明。
    我需要你写：
    1.Input Paragraph:小说的下一段。 输出段落应包含大约 20 个句子，并应遵循输入说明。
    2.Input Memory:更新后的内存。
    你应该首先解释输入记忆中哪些句子不再需要以及为什么，然后解释需要将哪些句子添加到记忆中以及为什么。
    之后你应该写入更新的内存。 除了您之前认为应该删除或添加的部分外，更新后的记忆应该与输入记忆相似。
    更新后的内存应该只存储关键信息。 更新后的记忆永远不要超过 20 句话！
    3.Input Instruction:下一步要写什么的指令（在你写完之后）。 你应该输出 3 条不同的指令，每条指令都是故事的一个可能有趣的延续。
    每个输出指令应包含大约 5 个句子
    以下是输入： 

    Input Memory:  
    {self.short_memory}

    Input Paragraph:
    {input_paragraph}

    Input Instruction:
    {input_instruction}

    Input Related Paragraphs:
    {input_long_term_memory}
    
    现在开始编写，严格按照以下输出格式组织输出：
    Output Paragraph: 
    <string of output paragraph>，大约20个句子。

    Output Memory: 
    Rational: <string that explain how to update the memory>;
    Updated Memory: <string of updated memory>，大约10到20个句子

    Output Instruction: 
    Instruction 1: <content for instruction 1>，5句左右
    Instruction 2: <content for instruction 2>，5句左右
    Instruction 3: <content for instruction 3>，5句左右

    很重要！！ 更新后的内存应该只存储关键信息。 更新后的记忆不得超过 500 个单词！
    最后，记住你正在写小说。 像小说家一样写作。
    请记住，每个章节将包含超过10个段落，而这本小说将包含超过100个章节。而这只是开始。请写一些接下来将发生的有趣情节。
    同时，在撰写输出指南时，请考虑什么情节对普通读者具有吸引力。

    很重要：
    你应该首先解释输入记忆中哪些句子不再需要以及为什么，然后解释需要将哪些句子添加到记忆中以及为什么。 之后，您开始重写输入内存以获得更新的内存。
    {new_character_prompt}
    """
        return input_text

    # 解析输出，用于更新记忆
    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b(
                "Output Paragraph:", "Output Memory", output
            )
            output_memory_updated = get_content_between_a_b(
                "Updated Memory:", "Output Instruction:", output
            )
            self.short_memory = output_memory_updated
            ins_1 = get_content_between_a_b("Instruction 1:", "Instruction 2", output)
            ins_2 = get_content_between_a_b("Instruction 2:", "Instruction 3", output)
            lines = output.splitlines()
            # Instruction 3的内容可能和I3在同一行也可能在下一行
            if lines[-1] != "\n" and lines[-1].startswith("Instruction 3"):
                ins_3 = lines[-1][len("Instruction 3:") :]
            elif lines[-1] != "\n":
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,
                "output_paragraph": output_paragraph,
                "output_instruction": [
                    instruction.strip() for instruction in output_instructions
                ],
            }

            return output
        except:
            return None

    def step(self):
        # 准备输入文本
        prompt = self.prepare_input()

        # 打印输入文本
        print(prompt + "\n" + "\n")

        # 获取API响应
        response = get_api_response(prompt)

        # 解析API响应并存储到output属性中
        self.output = self.parse_output(response)

        # 如果解析失败，则重新获取API响应，直到成功解析为止
        while self.output == None:
            response = get_api_response(prompt)
            self.output = self.parse_output(response)

        # 将当前输出段落添加到长期记忆中
        self.long_memory.append(self.input["output_paragraph"])

        # 使用文本嵌入器对长期记忆进行编码，并更新记忆索引
        self.memory_index = self.embedder.encode(
            self.long_memory, convert_to_tensor=True
        )

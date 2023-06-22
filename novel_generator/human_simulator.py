
from utils import get_content_between_a_b, parse_instructions,get_api_response

class Human:

    def __init__(self, input, memory, embedder):
        self.input = input
        if memory:
            self.memory = memory
        else:
            self.memory = self.input['output_memory']
        self.embedder = embedder
        self.output = {}

    """
    生成输入文本的格式，包括之前写过的段落、ChatGPT助手写的新段落、故事情节总结和下一步写作计划。返回生成的输入文本。
    """
    def prepare_input(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        user_edited_plan = self.input["output_instruction"]

        input_text = f"""
        现在想象你是一位小说家，在 ChatGPT 的帮助下写了一本中国小说。
        你会得到一个之前写好的段落（你写的），和一个你的 ChatGPT 助手写的段落，一个由你的 ChatGPT 助手维护的主要故事情节的总结，以及一个由你的 ChatGPT 助手提出的下一步写什么的计划。
    我需要你写：
    1. Extended Paragraph:将ChatGPT助手编写的新段落扩展为您ChatGPT助手编写的段落长度的两倍。
    2. Selected Plan:复制ChatGPT助手提出的方案。
    3. Revised Plan:将选定的计划修订为下一段的大纲。
    
    Previously written paragraph:  
    {previous_paragraph}

    The summary of the main storyline maintained by your ChatGPT assistant:
    {memory}

    The new paragraph written by your ChatGPT assistant:
    {writer_new_paragraph}

    The plan of what to write next proposed by your ChatGPT assistant:
    {user_edited_plan}

    现在开始编写，严格按照如下输出格式组织输出,所有输出仍然保持是中文:
    
    Extended Paragraph: 
    <string of output paragraph>，大约40-50个句子。

    Selected Plan: 
    <copy the plan here>

    Revised Plan:
    <string of revised plan>，尽量简短，5-7句左右。

    很重要：
    请记住，您正在写小说。 像小说家一样写作，在写下一段的计划时不要动作太快。 在选择和扩展计划时，考虑该计划如何吸引普通读者。 请记住遵守长度限制！ 请记住，章节将包含 10 多个段落，而小说将包含 100 多个章节。 下一段将是第二章的第二段。 您需要为以后的故事留出空间。

    """
        return input_text
    
    # 从生成的文本中提取选定的计划
    def parse_plan(self,response):
        plan = get_content_between_a_b('Selected Plan:','Reason',response)
        return plan

    # 用于选择最有趣和最合适的计划。它生成选择计划的格式化提示，并通过调用get_api_response方法获取模型生成的回复。然后使用parse_plan方法解析回复中的选定计划。
    def select_plan(self,response_file):
        
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        previous_plans = self.input["output_instruction"]
        prompt = f"""
    现在想象你是一个有用的助手，帮助小说家做决定。
    您将获得一段之前写过的段落和一段由 ChatGPT 写作助手编写的段落，由 ChatGPT 助手维护的主要故事情节摘要，以及 3 种不同的可能的下一步写作计划。
    我需要你：
    选择 ChatGPT 助手建议的最有趣和最合适的计划。

    Previously written paragraph:  
    {previous_paragraph}

    The summary of the main storyline maintained by your ChatGPT assistant:
    {memory}

    The new paragraph written by your ChatGPT assistant:
    {writer_new_paragraph}

    Three plans of what to write next proposed by your ChatGPT assistant:
    {parse_instructions(previous_plans)}

    现在开始选择，严格按照以下输出格式组织输出：
      
    Selected Plan: 
    <copy the selected plan here>

    Reason:
    <Explain why you choose the plan>
    """
        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)

        plan = self.parse_plan(response)
        while plan == None:
            response = get_api_response(prompt)
            plan= self.parse_plan(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Selected plan here:\n{response}\n\n")

        return plan
    
    # 用于解析生成的输出文本。它从文本中提取扩展段落和修订计划，并将它们存储在字典中。
    def parse_output(self, text):
        try:
            if text.splitlines()[0].startswith('Extended Paragraph'):
                new_paragraph = get_content_between_a_b(
                    'Extended Paragraph:', 'Selected Plan', text)
            else:
                new_paragraph = text.splitlines()[0]

            lines = text.splitlines()
            if lines[-1] != '\n' and lines[-1].startswith('Revised Plan:'):
                revised_plan = lines[-1][len("Revised Plan:"):]
            elif lines[-1] != '\n':
                revised_plan = lines[-1]

            output = {
                "output_paragraph": new_paragraph,
                # "selected_plan": selected_plan,
                "output_instruction": revised_plan,
                # "memory":self.input["output_memory"]
            }

            return output
        except:
            return None

    """
    执行一步生成任务的方法。它调用prepare_input方法生成输入文本，并通过调用get_api_response方法获取模型生成的回复。
    然后使用parse_output方法解析回复中的输出。如果解析失败，则继续获取模型生成的回复，直到成功解析输出或达到最大尝试次数。
    """
    def step(self):

        prompt = self.prepare_input()
        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)
        self.output = self.parse_output(response)
        while self.output == None:
            response = get_api_response(prompt)
            self.output = self.parse_output(response)

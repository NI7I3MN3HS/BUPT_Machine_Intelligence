import re
import openai

openai.api_key="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"

# 用于向 OpenAI GPT 模型发送请求并获取生成的文本响应
def get_api_response(content: str, max_tokens=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "你是一个乐于助人且富有创造力的小说写作助手。请使用中文回答。",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )

    return response["choices"][0]["message"]["content"]


# 用于从给定的文本 text 中提取位于字符串 a 和字符串 b 之间的内容
def get_content_between_a_b(a, b, text):
    return re.search(f"{a}(.*?)\n{b}", text, re.DOTALL).group(1).strip()

def get_init(text=None):
    """
    根据给定的文本 text 生成初始段落和指令

    text: 可选参数，作为初始化提示生成内容的文本
    返回包含初始段落和指令的字典
    """
    response = get_api_response(text)
    print(response)

    paragraphs = {
        "name": "",
        "Outline": "",
        "Paragraph 1": "",
        "Paragraph 2": "",
        "Paragraph 3": "",
        "Summary": "",
        "Instruction 1": "",
        "Instruction 2": "",
        "Instruction 3": "",
    }

    paragraphs["name"] = get_content_between_a_b("Name:", "Outline", response)

    paragraphs["Paragraph 1"] = get_content_between_a_b(
        "Paragraph 1:", "Paragraph 2:", response
    )
    paragraphs["Paragraph 2"] = get_content_between_a_b(
        "Paragraph 2:", "Paragraph 3:", response
    )
    paragraphs["Paragraph 3"] = get_content_between_a_b(
        "Paragraph 3:", "Summary", response
    )
    paragraphs["Summary"] = get_content_between_a_b(
        "Summary:", "Instruction 1", response
    )
    paragraphs["Instruction 1"] = get_content_between_a_b(
        "Instruction 1:", "Instruction 2", response
    )
    paragraphs["Instruction 2"] = get_content_between_a_b(
        "Instruction 2:", "Instruction 3", response
    )

    lines = response.splitlines()
    # Instruction 3的内容可能和I3在同一行也可能在下一行
    if lines[-1] != "\n" and lines[-1].startswith("Instruction 3"):
        paragraphs["Instruction 3"] = lines[-1][len("Instruction 3:") :]
    elif lines[-1] != "\n":
        paragraphs["Instruction 3"] = lines[-1]
    # 有时它给出章节大纲，有时它不给出章节大纲
    for line in lines:
        if line.startswith("Chapter"):
            paragraphs["Outline"] = get_content_between_a_b(
                "Outline:", "Chapter", response
            )
            break
    if paragraphs["Outline"] == "":
        paragraphs["Outline"] = get_content_between_a_b(
            "Outline:", "Paragraph", response
        )

    return paragraphs


# 使用 ChatGPT 模型生成对话回复。它通过提供的 prompt 来与模型进行交互，并返回模型生成的回复
def get_chatgpt_response(model, prompt):
    response = ""
    for data in model.ask(prompt):
        response = data["message"]
    model.delete_conversation(model.conversation_id)
    model.reset_chat()
    return response


# 用于解析指令列表并生成一个格式化的字符串输出
def parse_instructions(instructions):
    output = ""
    for i in range(len(instructions)):
        output += f"{i+1}. {instructions[i]}\n"
    return output

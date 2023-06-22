import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_init, parse_instructions
import re

_CACHE = {}

# 构建语义搜索模型
embedder = SentenceTransformer("multi-qa-mpnet-base-cos-v1")


# 用于生成初始化提示
def init_prompt(novel_type, description):
    if description == "":
        description = ""
    else:
        description = " about " + description
    return f"""
请写一部 50 章的{novel_type}小说{description}。 严格遵循以下格式：
从小说的名字开始。
接下来，写出第一章的大纲。 大纲应该描述小说的背景和开头。
根据你的大纲，写出前三段对小说的描述。 以小说风格写作，花时间设置场景。以小说风格写作，花时间设置场景。
写一个总结，抓住三个段落的关键信息。
最后，为接下来要写的内容写三个不同的说明，每个说明大约包含五个句子。 每条指令都应该呈现故事的一个可能的、有趣的延续。
输出格式应遵循以下准则：
Name: <name of the novel>
Outline: <outline for the first chapter>   
Paragraph 1: <content for paragraph 1>
Paragraph 2: <content for paragraph 2>
Paragraph 3: <content for paragraph 3>
Summary: <content of summary>> 
Instruction 1: <content for instruction 1>
Instruction 2: <content for instruction 2>
Instruction 3: <content for instruction 3>

确保精确并严格遵循输出格式,并使用中文。
"""


def init(novel_type, description, request: gr.Request):
    if novel_type == "":
        novel_type = "Science Fiction"
    global _CACHE
    # 准备第一个初始化
    init_paragraphs = get_init(text=init_prompt(novel_type, description))
    # print(init_paragraphs)
    start_input_to_human = {
        "output_paragraph": init_paragraphs["Paragraph 3"],
        "input_paragraph": "\n\n".join(
            [init_paragraphs["Paragraph 1"], init_paragraphs["Paragraph 2"]]
        ),
        "output_memory": init_paragraphs["Summary"],
        "output_instruction": [
            init_paragraphs["Instruction 1"],
            init_paragraphs["Instruction 2"],
            init_paragraphs["Instruction 3"],
        ],
    }

    _CACHE["cookie"] = {
        "start_input_to_human": start_input_to_human,
        "init_paragraphs": init_paragraphs,
    }
    written_paras = f"""Title: {init_paragraphs['name']}

Outline: {init_paragraphs['Outline']}

Paragraphs:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions(
        [init_paragraphs["Paragraph 1"], init_paragraphs["Paragraph 2"]]
    )
    # short memory, long memory, current written paragraphs, 3 next instructions
    return (
        start_input_to_human["output_memory"],
        long_memory,
        written_paras,
        init_paragraphs["Instruction 1"],
        init_paragraphs["Instruction 2"],
        init_paragraphs["Instruction 3"],
    )


def step(
    short_memory,
    long_memory,
    instruction1,
    instruction2,
    instruction3,
    current_paras,
    request: gr.Request,
):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    cache = _CACHE["cookie"]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human["output_instruction"] = [
            instruction1,
            instruction2,
            instruction3,
        ]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human, memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs["Summary"]
        writer_start_input = human.output

        # 初始化Wirter
        writer = RecurrentGPT(
            input=writer_start_input,
            short_memory=start_short_memory,
            long_memory=[
                init_paragraphs["Paragraph 1"],
                init_paragraphs["Paragraph 2"],
            ],
            memory_index=None,
            embedder=embedder,
        )
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output["output_memory"] = short_memory
        # 从三个指令中随机选择一个指令
        instruction_index = random.randint(0, 2)
        output["output_instruction"] = [instruction1, instruction2, instruction3][
            instruction_index
        ]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # 短期记忆，长期记忆，当前写的段落，3个下一步的指令
    return (
        writer.output["output_memory"],
        long_memory,
        current_paras + "\n\n" + writer.output["input_paragraph"],
        human.output["output_instruction"],
        *writer.output["output_instruction"],
    )


def controled_step(
    short_memory,
    long_memory,
    selected_instruction,
    current_paras,
    request: gr.Request,
):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    cache = _CACHE["cookie"]
    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human["output_instruction"] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human, memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs["Summary"]
        writer_start_input = human.output

        # 初始化Wirter
        writer = RecurrentGPT(
            input=writer_start_input,
            short_memory=start_short_memory,
            long_memory=[
                init_paragraphs["Paragraph 1"],
                init_paragraphs["Paragraph 2"],
            ],
            memory_index=None,
            embedder=embedder,
        )
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output["output_memory"] = short_memory
        output["output_instruction"] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    # 短期记忆，长期记忆，当前写的段落，3个下一步的指令
    return (
        writer.output["output_memory"],
        parse_instructions(writer.long_memory),
        current_paras + "\n\n" + writer.output["input_paragraph"],
        *writer.output["output_instruction"],
    )


# 选择下一步的指令
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan - 1]
    return selected_plan


with gr.Blocks(
    title="AI小说生成器", css="footer {visibility: hidden}", theme="default"
) as demo:
    gr.Markdown(
        """
    # 基于GPT的自动小说生成器
    
    可以根据题目和简介自动续写文章
    
    也可以手动选择剧情走向进行续写 

    跑的比较慢，需要等待一段时间才能出结果
    """
    )
    with gr.Tab("自动剧情"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="请输入文本", placeholder="可以自己填写或者从EXamples中选择一个填入"
                            )
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="剧情简介（非必选项）")
                btn_init = gr.Button("点击开始运行", variant="primary")
                gr.Examples(
                    [
                        "科幻小说",
                        "爱情小说",
                        "推理小说",
                        "奇幻小说",
                        "玄幻小说",
                        "恐怖",
                        "悬疑",
                        "惊悚",
                        "武侠小说",
                    ],
                    inputs=[novel_type],
                )
                written_paras = gr.Textbox(label="文章内容 (可编辑)", max_lines=21, lines=21)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### 剧情模式\n")
                    short_memory = gr.Textbox(label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(label="长期记忆 (可编辑)", max_lines=6, lines=6)
                    # long_memory = gr.Dataframe(
                    #     # label="Long-Term Memory (editable)",
                    #     headers=["Long-Term Memory (editable)"],
                    #     datatype=["str"],
                    #     row_count=3,
                    #     max_rows=3,
                    #     col_count=(1, "fixed"),
                    #     type="array",
                    # )
                with gr.Box():
                    gr.Markdown("### 选项模型\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="指令1(可编辑)", max_lines=4, lines=4
                        )
                        instruction2 = gr.Textbox(
                            label="指令2(可编辑)", max_lines=4, lines=4
                        )
                        instruction3 = gr.Textbox(
                            label="指令3(可编辑)", max_lines=4, lines=4
                        )
                    selected_plan = gr.Textbox(
                        label="选项说明 (来自上一步)", max_lines=2, lines=2
                    )

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(
            init,
            inputs=[novel_type, description],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
            ],
        )
        btn_step.click(
            step,
            inputs=[
                short_memory,
                long_memory,
                instruction1,
                instruction2,
                instruction3,
                written_paras,
            ],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                selected_plan,
                instruction1,
                instruction2,
                instruction3,
            ],
        )

    with gr.Tab("自选择剧情"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="请输入文本", placeholder="可以自己填写或者从EXamples中选择一个填入"
                            )
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="剧情简介（非必选项）")
                btn_init = gr.Button("点击开始运行", variant="primary")
                gr.Examples(
                    [
                        "科幻小说",
                        "爱情小说",
                        "推理小说",
                        "奇幻小说",
                        "玄幻小说",
                        "恐怖",
                        "悬疑",
                        "惊悚",
                        "武侠小说",
                    ],
                    inputs=[novel_type],
                )
                written_paras = gr.Textbox(label="文章内容 (可编辑)", max_lines=23, lines=23)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### 剧情模型\n")
                    short_memory = gr.Textbox(label="短期记忆 (可编辑)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(label="长期记忆 (可编辑)", max_lines=6, lines=6)
                with gr.Box():
                    gr.Markdown("### 选项模型\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="指令1", max_lines=3, lines=3, interactive=False
                        )
                        instruction2 = gr.Textbox(
                            label="指令2", max_lines=3, lines=3, interactive=False
                        )
                        instruction3 = gr.Textbox(
                            label="指令3", max_lines=3, lines=3, interactive=False
                        )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(
                                ["Instruction 1", "Instruction 2", "Instruction 3"],
                                label="指令 选择",
                            )
                            #  info="Select the instruction you want to revise and use for the next step generation.")
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="在上一步骤中被选择的 (可编辑)", max_lines=5, lines=5
                            )

                btn_step = gr.Button("下一步", variant="primary")

        btn_init.click(
            init,
            inputs=[novel_type, description],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
            ],
        )
        btn_step.click(
            controled_step,
            inputs=[short_memory, long_memory, selected_instruction, written_paras],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
            ],
        )
        selected_plan.select(
            on_select,
            inputs=[instruction1, instruction2, instruction3],
            outputs=[selected_instruction],
        )

    demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch()

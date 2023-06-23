## 1. 如何运行

### 1.1 运行环境

> python>=3.9
> gradio
> sentence_transformers
> openai

> 安装相应库的命令：

```shell
pip install gradio
pip install openai
pip install sentence_transformers
```

### 1.2 本地运行

> 1、进入 utils.py 文件填写你自己的 OPENAI_API_KEY

```utils.py line:4
openai.api_key="YOUR_OPENAI_API_KEY"
```

> 2、终端运行如下命令：

```sh
python gradio_server.py
```

> 3、浏览器打开：[Demo](http://127.0.0.1:7860)

## 2. 项目结构

```
├── 1.png
├── 2.png
├── 3.png
├── README.md
├── gradio_server.py
├── human_simulator.py
├── recurrentgpt.py
├── utils.py
├── 实验报告.md
└── 实验报告.pdf
```

## 3. 其他事项

> 1、若运行错误，请先检查相应库和 OPENAI_API_KEY 是否正确设置
>
> 2、由于网络原因和大模型的特点，生成小说文本的时间较长，需要耐心等待
>
> 3、项目中自带了一个 API_KEY 和备用 API_KEY，随时可能用完，如果用完请尝试填写自己购买的 API_KEY

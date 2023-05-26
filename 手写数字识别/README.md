## 1. 如何运行

### 1.1 运行环境

> python>=3.9 flask pytorch

### 1.2 运行

```sh



# 在根目录下运行如下命令



python main.py



```

> 若运行出错，可以尝试先运行以下代码，再重新运行

```sh



# 在根目录下运行如下命令



python train/cnn.py & python train/bp.py



```

> flask 默认端口为 5000 端口，macos 下 airplay 会占用 5000 端口，需要关闭 airplay 或者在 main.py 中修改端口号，如下：

```python

# main.py

if __name__ == "__main__":

app.run(host="0.0.0.0", port=8000)

```

浏览器打开：[Demo(默认 5000 端口)](http://127.0.0.1:5000)

## 2. 项目结构

> 1. 请确保 bp.ckpt 和 cnn.ckpt 与 main.py 在同一目录下
> 2. data 文件夹中 为 mnist 数据集
> 3. static 文件夹 和 templates 文件夹中为前端文件

```



├── README.md



├── bp.ckpt



├── cnn.ckpt



├── data



│ └── MNIST



│ └── raw



│ ├── t10k-images-idx3-ubyte



│ ├── t10k-images-idx3-ubyte.gz



│ ├── t10k-labels-idx1-ubyte



│ ├── t10k-labels-idx1-ubyte.gz



│ ├── train-images-idx3-ubyte



│ ├── train-images-idx3-ubyte.gz



│ ├── train-labels-idx1-ubyte



│ └── train-labels-idx1-ubyte.gz



├── main.py



├── static



│ ├── css



│ │ └── bootstrap.min.css



│ └── js



│ ├── jquery.min.js



│ └── main.js



├── templates



│ └── index.html



├── train



│ ├── bp.py



│ └── cnn.py



├── 实验报告.md



└── 实验报告.pdf



```

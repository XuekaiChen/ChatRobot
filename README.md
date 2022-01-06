# 基于seq2seq的闲聊系统

效果：
```bash
> hello?
Bot: hello .
> where am I?
Bot: you re in a hospital .
> who are you?
Bot: i m a lawyer .
> how are you doing?
Bot: i m fine .
> are you my friend?
Bot: no .
> you're under arrest
Bot: i m trying to help you !
> i'm just kidding
Bot: i m sorry .
> where are you from?
Bot: san francisco .
> it's time for me to leave
Bot: i know .
> goodbye
Bot: goodbye .
```

## 1. 准备数据
通常情况下，多轮对话的场景不多，因此项目考虑使用康奈尔电影对话语料库[Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 数据集进行模型的训练。下载完数据后，在根目录下创建一个data文件夹，将解压后的Cornell Movie-Dialogs Corpus文件夹放在里面。

## 2. 数据预处理
`preprocessing.ipynb`交互式查看处理状态，其代码在`utils/file_utils.py`中也有

## 3. 模型训练和评估
运行`main.py`函数，可适当调整`args`中的参数值，如max_length，learning_rate等
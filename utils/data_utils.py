# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : load data and data cleaning

import os
import re
import unicodedata
import pickle
from utils.vocab import Vocab

MAX_LENGTH = 10  # Maximum sentence length to consider


def unicode2ascii(s):
    # 把 Unicode 字符串转换成 纯ASCII字符串，参考：
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def read_pairs(datafile):
    """read_pairs，从转换好的 datafile 文件中读取对话文本对。
    """
    # Read query/response pairs
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    return pairs


def filter_pair(p):
    """Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold"""
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """filter_pairs
    使用 filter_pair 函数来过滤掉短句子
    """
    # Filter pairs using filter_pair condition
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus_name, datafile, save_dir=None):
    """Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
    Args:
        corpus_name: 语料集名称，如："cornell movie-dialogs corpus"。
        datafile: 转换好的语料集文件的路径。
        save_dir: 可选，默认为None，保存在硬盘中。
    Return:
        vocab: 单词词典。
        pairs: 读取的对话文本对。
    """

    ################################################################
    # TODO 可选，这里的 load_prepare_data 运行一次所需时间较长，我们可以在这里写一段 save and restore 的代码，处理好的数据就存在硬盘里，直接读取，节省再次处理的时间，否则，就进行处理，然后保存
    if save_dir:
        with open("data\\cornell movie-dialogs corpus\\vocab.pkl", "rb") as f_vocab:
            vocab = pickle.load(f_vocab)
        with open("data\\cornell movie-dialogs corpus\\pairs.pkl", "rb") as f_pairs:
            pairs = pickle.load(f_pairs)
        print("已获取大小为{!s}的vocab,长度为{!s}的pairs".format(len(vocab.word2count), len(pairs)))
        return vocab, pairs

    ################################################################
    else:
        print("开始准备 vocab， pairs 数据 ...")
        vocab = Vocab(corpus_name)  # 创建一个空的 vocab 对象
        pairs = read_pairs(datafile)
        print("Read {!s} sentence pairs".format(len(pairs)))

        pairs = filter_pairs(pairs)  # 过滤掉短句子
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:  # 向vocab中添加单词
            vocab.add_sentence(pair[0])
            vocab.add_sentence(pair[1])
        print("Counted words:", vocab.num_words)
        # 存储vocab.pkl和pairs.pkl
        with open("data/cornell movie-dialogs corpus/vocab.pkl", 'wb') as f_vocab:
            pickle.dump(vocab, f_vocab)
        with open("data/cornell movie-dialogs corpus/pairs.pkl", 'wb') as f_pairs:
            pickle.dump(pairs, f_pairs)

        return vocab, pairs

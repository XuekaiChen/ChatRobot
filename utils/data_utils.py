# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : load data and data cleaning

import os
import re
import unicodedata
from solutions.utils.vocab import Vocab

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
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    return pairs


def filter_pair(p):
    """Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold"""
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """filter_pairs
    使用 filter_pair 函数来
    """
    # Filter pairs using filter_pair condition
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus_name, datafile, save_dir=None):
    """Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    Args:
        corpus_name: 语料集名称，如："cornell movie-dialogs corpus"。
        datafile: 转换好的语料集文件的路径。
        save_dir: 可选，默认为None，保存在硬盘的路径。
    Return:
        vocab: 单词词典。
        pairs: 读取的对话文本对。
    """

    import pickle
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    pairs_path = os.path.join(save_dir, "pairs.pkl")

    if os.path.isfile(vocab_path) and os.path.isfile(pairs_path):
        print("恢复已经保存的 vocab， pairs 数据 ...")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f, encoding="bytes")
        with open(pairs_path, "rb") as f:
            pairs = pickle.load(f, encoding="bytes")

        return vocab, pairs

    print("开始准备 vocab， pairs 数据 ...")
    vocab = Vocab(corpus_name)  # 创建一个 vocab 对象
    pairs = read_pairs(datafile)
    print("Read {!s} sentence pairs".format(len(pairs)))

    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])
    print("Counted words:", vocab.num_words)

    print("Saving vocab to: {vocab_path}")
    os.makedirs(save_dir, exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("Saving pairs to: {pairs_path}")
    with open(pairs_path, "wb") as f:
        pickle.dump(pairs, f)

    return vocab, pairs

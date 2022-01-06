# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : adjust text into train input

import os
import itertools
import random
import torch
from utils.data_utils import load_prepare_data
from utils.vocab import PAD_token, EOS_token


def indices_from_sentence(vocab, sentence):
    """indices_from_sentence，根据字典 vocab，从句子 sentence 中
    Input:

    Return:
        一个 Python 列表，按顺序包含 sentence 中各个单词的索引，最后加上 EOS_token 符。
    """
    ################################################################
    # TODO 返回 sentence 中各个单词的索引列表，注意最后加上 EOS_token。
    # 提示：用 vocab.word2index 属性检索 word 对应的索引
    return [vocab.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    ################################################################


def zero_padding(l, fillvalue=PAD_token):
    """
        zero_padding, 把 l 中的所有向量补 PAD_token 即补零，并把 l 转置，参见：https://docs.python.org/3/library/itertools.html
        转置的作用：batch[1]中所有内容为打头的单词（即timestep[1]），以此类推
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value=PAD_token):
    """
    Input:
        l: 一组句子索引序列构成的列表
        value: 用来判断的值 
    Return:
        mask: 返回的一个属于嵌套列表结构的mask矩阵，每个元素mask[i][j]表示 l的第i个句子的第j个元素 l[i][j] 是否是 value
    """
    ################################################################
    # TODO 构造 mask，一 Python 嵌套列表
    # 对于 mask 中的每一个 list，遍历 l 中对应的序列 seq，如果遍历到的 token 是 value（即PAD_token），把 mask[i][j] 置为 0，否则置为1
    mask = []
    for i, seq in enumerate(l):
        mask.append([])
        for token in seq:
            if token == value:
                mask[i].append(0)
            else:
                mask[i].append(1)
    return mask
    ################################################################


def input_variable(l, vocab):
    """
    Input:
        l: batch of word list
        vocab: for word2index
    Return:
         padded_var: padded input sentence
         lengths: padded input sentence length
    """
    ################################################################
    # TODO step 1：使用 indices_from_sentence 把 l 变成此batch的单词索引
    indices_batch = [indices_from_sentence(vocab, sentence) for sentence in l]
    ################################################################
    lengths = torch.tensor([len(indexes) for indexes in indices_batch])
    padded_list = zero_padding(indices_batch)

    ################################################################
    # TODO step 2：使用 torch.LongTensor 方法把 padded_list 转换成长整型 tensor
    padded_var = torch.LongTensor(padded_list)
    ################################################################

    return padded_var, lengths


def output_variable(l, vocab):
    ################################################################
    # TODO step 1：使用 indices_from_sentence 把 l 变成此batch的单词索引
    indices_batch = [indices_from_sentence(vocab, sentence) for sentence in l]
    ################################################################

    max_target_len = max([len(indexes) for indexes in indices_batch])
    padded_list = zero_padding(indices_batch)

    # TODO 首先完成 binary_matrix 函数
    mask = binary_matrix(padded_list)

    ################################################################
    # TODO step 2：
    # 使用 torch.BoolTensor 方法把 mask 转换成布尔型 tensor
    # 使用 torch.LongTensor 方法把 padded_list 转换成长整型 tensor
    mask = torch.BoolTensor(mask)
    padded_var = torch.LongTensor(padded_list)
    ################################################################

    return padded_var, mask, max_target_len


def batch_to_train_data(vocab, pair_batch):
    """batch_to_train_data, 构建一个 batch 的 train tensor 数据
    Input:
        vocab:
        pair_batch:
    Return:
        input_:
        lengths:
        output:
        mask: 
        max_target_len:
    """
    # 排序
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    # 得到原始的输入batch，和对应的原始的输出batch
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    ################################################################
    # TODO step 1 使用 input_variable 函数接口获得 input_, lengths tensor
    ################################################################
    input_, lengths = input_variable(input_batch, vocab)
    ################################################################
    # TODO step 2 使用 output_variable 函数接口获得 output, mask, max_target_len tensor
    ################################################################
    output, mask, max_target_len = output_variable(output_batch, vocab)

    # 按顺序返回所有这 5个 值
    return input_, lengths, output, mask, max_target_len


MIN_COUNT = 3  # Minimum word count threshold for trimming


def trim_rare_words(vocab, pairs, MIN_COUNT):
    # 根据 MIN_COUNT 剔除掉 vocab 中频次低的单词
    vocab.trim(MIN_COUNT)
    # 根据剔除好的 vocab 过滤 句子对
    keep_pairs = []
    for pair in pairs:
        input_sentence, output_sentence = pair[0], pair[1]
        keep_input, keep_output = True, True
        ################################################################
        # TODO 检查输入句子中是否有稀少单词，如果有则不保留
        for word in input_sentence.split(' '):
            if word not in vocab.word2index:
                keep_input = False
                break
        ################################################################
        ################################################################
        # TODO 检查输出句子，思路与上面完全一致
        for word in output_sentence.split(' '):
            if word not in vocab.word2index:
                keep_output = False
                break
        ################################################################
        # 只有对话对中的两个句子都没有稀少单词时，才保留对话对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def main():
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    # Load/Assemble vocab and pairs
    save_dir = os.path.join("data", "save")
    vocab, pairs = load_prepare_data(corpus_name, datafile, save_dir)

    # Trim vocab and pairs
    pairs = trim_rare_words(vocab, pairs, MIN_COUNT)  # 若有罕见词，则直接删除pairs

    # Example for validation
    small_batch_size = 5
    batches = batch_to_train_data(vocab, [random.choice(pairs)
                                          for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)


if __name__ == "__main__":
    main()

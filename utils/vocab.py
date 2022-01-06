# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : construct dictionary of word2index and index2word from text

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocab(object):

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",  # 补白标志
                           SOS_token: "SOS",  # 句子开始标志
                           EOS_token: "EOS"}  # 句子结束标志
        self.num_words = 3  # 数单词个数的时候把 SOS, EOS, PAD 算上

    def add_sentence(self, sentence):
        """用空格拆分句子字符串，得到单词，使用 add_word 把所有单词添加到 Vocab 中。
        """
        words = sentence.split()
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            word: The token to be added.
        """
        ################################################################
        # TODO: add_word function
        # 输入：word；
        # 输出：无
        # 涉及的成员变量：word2index，word2count，index2word，num_words
        # word2index 是一个 Python 字典，本函数需要判断输入单词 word 是否在 word2index 的键（Key）中。如果不在，则用当前的 num_words 值对其进行编号，赋这个键对于的值为 num_words，并让字典 word2count word键上的值初始化为 1。word2count字典用来对 word 出现的频次进行计数。如果 word 已经在 word2index 中，则 word 键对应的word2count 频次加一。
        # 根据注释完成本函数。
        if word in self.word2index.keys():
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        ################################################################

    def trim(self, min_count):
        """trim 根据特定的频次阈值，删除频次不高的单词
        Input:
            min_count: 频次阈值
        """
        if self.trimmed:  # 已经完成好了剔除的话则不需要再次剔除
            return
        self.trimmed = True  # 已经完成单词剔除

        keep_words = []
        ################################################################
        # TODO 根据 word2count 字典提供的每个单词的频次（对应键的值），剔除掉频次小于 min_count 的单词，提示：keep_words 变量可以用来存储需要保存的单词列表
        for k, v in self.word2count.items():
            if v >= min_count:  # 注意这里要循环遍历word2count中元素，因此低频次不可以直接pop()掉
                keep_words.append(k)

        # 重新构建词表
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS",
                           EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.add_word(word)  # 但这种方法重新构建后word2count全变为1，不再有意义
        ################################################################



    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)


def main():
    vocab = Vocab("experiment")
    vocab.add_sentence("to be or not to be, this is a question")
    print(vocab.word2count)
    vocab.trim(2)
    print(vocab.word2count)


if __name__ == "__main__":
    main()

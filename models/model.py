# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : Model design. EncoderRNN, Attention mechanism and LuongAttentionDecoderRNN in the paper

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, num_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding  # 已经nn.embedding()实例化过的

        ################################################################
        # TODO 构建一个 nn.GRU 层，把 GRU 的 input_size 参数和 hidden_size 参数都设置为 'hidden_size'
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=(0 if num_layers == 1 else dropout),
                          bidirectional=True)
        ################################################################


    def forward(self, input_seq, input_lengths, hidden=None):
        """EncoderRNN 的 forward 函数
        """
        ################################################################
        # TODO step 1, 把 input_seq 转换成 embedded 变量
        ################################################################
        # NLP 中的 embedding 层，相当于 CNN 的全连接层。
        embedded = self.embedding(input_seq)

        ################################################################
        # TODO step 2, 把 embedded 打包，并把 packed以及hidden通过gru
        ################################################################
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)

        ################################################################
        # TODO step 3, 解包 outputs
        ################################################################
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # 将双向 GRU 的输出求和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # 返回 output 和最终的隐藏状态 hidden state

        return outputs, hidden


class Attention(nn.Module):
    # Luong attention layer
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))  # TODO

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attention(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attention(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttentionDecoderRNN(nn.Module):
    """Decoder with Luong-style Attention
    """

    def __init__(self, attention_model, embedding, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(LuongAttentionDecoderRNN, self).__init__()

        # Keep for reference
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          dropout=(0 if num_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attention = Attention(attention_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        ################################################################
        # TODO step 1, 把embedded, last_hidden通过gru
        ################################################################
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # TODO step 2, 计算 attention 权重
        attn_weights = self.attention(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        # TODO step 3，根据论文 Luong 的 eq. 5，拼接RNN的输出和context变量
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # 根据 Luong eq. 6 预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        # 返回输出和最终隐藏状态
        return output, hidden


def main():
    attention = Attention("concat", hidden_size=500)


if __name__ == "__main__":
    main()

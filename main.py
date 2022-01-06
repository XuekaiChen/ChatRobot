# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : train and evaluate the seq2seq model with RNNEncoder and LuongAttentionRNNDecoder

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import torch.nn as nn
from torch import optim
from utils.data_utils import load_prepare_data, MAX_LENGTH
from utils.preprocessing import trim_rare_words, MIN_COUNT
from models.model import EncoderRNN, LuongAttentionDecoderRNN
from models.training import train
from models.evaluating import GreedySearchDecoder, evaluate_input

# 详细过程参考：https://pytorch.org/tutorials/beginner/chatbot_tutorial.html


class Namespace:
    """这个Namespace用于构造参数变量args的存储空间
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def train_model():
    # 载入数据
    corpus_name = "cornell movie-dialogs corpus"
    corpus_path = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus_path, "formatted_movie_lines.txt")

    # Load/Assemble vocab and pairs
    save_dir = os.path.join("data", "cornell movie-dialogs corpus")
    vocab, pairs = load_prepare_data(corpus_name, datafile, save_dir)

    # Trim voc and pairs
    pairs = trim_rare_words(vocab, pairs, MIN_COUNT)

    print("Counted words:", vocab.num_words)

    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    args = Namespace()
    # 设置超参数
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    args.update(batch_size=64,
                max_length=MAX_LENGTH,
                clip=50.0,
                teacher_forcing_ratio=1.0,
                device=device)

    # 设置模型参数
    args.update(model_name='chatbot_model',
                attn_model='dot',  # 'general', 'concat'
                hidden_size=500,
                encoder_n_layers=2,
                decoder_n_layers=2,
                dropout=0.1,
                save_dir="ckpt")  # 模型权重保存位置

    # 设置训练/优化参数
    args.update(learning_rate=0.0001,
                decoder_learning_ratio=5.0,
                n_iteration=4000,  # 迭代次数 4000
                print_every=1,
                save_every=500)  # 每 500 次迭代保存权重

    # Set checkpoint to load from; set to None if starting from scratch
    checkpoint_path = None
    directory = os.path.join(args.save_dir,
                             args.model_name,
                             corpus_name,
                             f"{args.encoder_n_layers}-{args.decoder_n_layers}_{args.hidden_size}")
    checkpoint_path = \
        os.path.join(directory, f'{args.n_iteration}_checkpoint.pt')

    # Load model if a checkpoint_path is provided
    if os.path.isfile(checkpoint_path):
        # If loading on same machine the model was trained on
        checkpoint = torch.load(checkpoint_path)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocab.__dict__ = checkpoint['vocab_dict']

    print('Building encoder and decoder ...')

    # Initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, args.hidden_size)
    if os.path.isfile(checkpoint_path):
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding,
                         args.encoder_n_layers, args.dropout)
    decoder = LuongAttentionDecoderRNN(
        args.attn_model, embedding, args.hidden_size, vocab.num_words, args.decoder_n_layers, args.dropout)

    # 恢复权重
    if os.path.isfile(checkpoint_path):
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio)
    if os.path.isfile(checkpoint_path):
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    if USE_CUDA:
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    train(vocab, pairs, encoder, decoder, encoder_optimizer,
          decoder_optimizer, embedding, corpus_name, checkpoint_path, args)

    # encoder, decoder 被 train 好了


def eval_model():
    # 载入数据
    corpus_name = "cornell movie-dialogs corpus"
    corpus_path = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus_path, "formatted_movie_lines.txt")

    # Load/Assemble vocab and pairs
    save_dir = os.path.join("data", "cornell movie-dialogs corpus")
    vocab, pairs = load_prepare_data(corpus_name, datafile, save_dir)

    # Trim voc and pairs
    pairs = trim_rare_words(vocab, pairs, MIN_COUNT)

    print("Counted words:", vocab.num_words)

    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    args = Namespace()
    # 设置超参数
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    args.update(batch_size=64,
                max_length=MAX_LENGTH,
                clip=50.0,
                teacher_forcing_ratio=1.0,
                device=device)

    # 设置模型参数
    args.update(model_name='chatbot_model',
                attn_model='dot',  # 'general', 'concat'
                hidden_size=500,
                encoder_n_layers=2,
                decoder_n_layers=2,
                dropout=0.1,
                save_dir="ckpt")  # 模型权重保存位置

    # 设置训练/优化参数
    args.update(learning_rate=0.0001,
                decoder_learning_ratio=5.0,
                n_iteration=4000,  # 迭代次数 4000
                print_every=1,
                save_every=500)  # 每 500 次迭代保存权重

    # Set checkpoint to load from; set to None if starting from scratch
    checkpoint_path = None
    directory = os.path.join(args.save_dir,
                             args.model_name,
                             corpus_name,
                             f"{args.encoder_n_layers}-{args.decoder_n_layers}_{args.hidden_size}")
    checkpoint_path = \
        os.path.join(directory, f'{args.n_iteration}_checkpoint.pt')

    # Load model if a checkpoint_path is provided
    if os.path.isfile(checkpoint_path):
        # If loading on same machine the model was trained on
        checkpoint = torch.load(checkpoint_path)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocab.__dict__ = checkpoint['vocab_dict']

    print('Building encoder and decoder ...')

    # Initialize word embeddings
    embedding = nn.Embedding(vocab.num_words, args.hidden_size)
    if os.path.isfile(checkpoint_path):
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding,
                         args.encoder_n_layers, args.dropout)
    decoder = LuongAttentionDecoderRNN(
        args.attn_model, embedding, args.hidden_size, vocab.num_words, args.decoder_n_layers, args.dropout)

    # 恢复权重
    if os.path.isfile(checkpoint_path):
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print('Models built and ready to go!')

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, device=device)

    # 取消注释来运行，聊天程序
    evaluate_input(encoder, decoder, searcher, vocab)


def main():
    train_model() # # 取消注释来运行 train_model()
    eval_model()  # 取消注释来运行 eval_model()


if __name__ == "__main__":
    main()

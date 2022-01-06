# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : use maskNLLLoss to train

import os
import random
import torch
import torch.nn as nn
from utils.data_utils import MAX_LENGTH
from utils.vocab import SOS_token
from utils.preprocessing import batch_to_train_data


def maskNLLLoss(input_, target, mask):
    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(input_, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(input_.device)
    return loss, nTotal.item()


def train_iteration(input_, lengths, target, mask, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, args):
    """
    Input:
        args: args should contain ```device```, ```teacher_forcing_ratio```, ```clip```
    Return:
    """
    device = args.device
    teacher_forcing_ratio = args.teacher_forcing_ratio
    clip = args.clip
    batch_size = args.batch_size

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_ = input_.to(device)
    target = target.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.num_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random(
    ) < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        if use_teacher_forcing:
            # Teacher forcing: next input is current target
            decoder_input = target[t].view(1, -1)
        else:
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
        # Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(
            decoder_output, target[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train(vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, corpus_name, checkpoint_name, args):
    """
    Input:
        args: args should contain ```device```, ```teacher_forcing_ratio```, ```clip```
    Return:
    """
    model_name = args.model_name
    hidden_size = args.hidden_size
    encoder_n_layers = args.encoder_n_layers
    decoder_n_layers = args.decoder_n_layers
    batch_size = args.batch_size
    n_iteration = args.n_iteration
    print_every = args.print_every
    save_every = args.save_every
    clip = args.clip

    # Load batches for each iteration
    # 每次从句子对 pairs 中随机选择 batch_size 个句子对，用 batch_to_train_data 函数处理成 training_batch tensor，循环做 n_iteration 次，构建模型训练所需的所有 training_batch 组成的列表 training_batches。
    training_batches = [batch_to_train_data(vocab, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if os.path.isfile(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_, lengths, target, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train_iteration(input_, lengths, target, mask, max_target_len,
                               encoder, decoder, encoder_optimizer, decoder_optimizer, args)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(args.save_dir, model_name, corpus_name,
                                     f'{encoder_n_layers}-{decoder_n_layers}_{hidden_size}')
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'vocab_dict': vocab.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, f'{iteration}_checkpoint.pt'))


def main():
    pass


if __name__ == "__main__":
    main()

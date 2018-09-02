"""Modules for building the char rnn"""

from torch import nn
from pytorch_utils.wrapped_lstm import WrappedLSTM

class CharDecoder(nn.Module):
    """
    Character level decoder LSTM to generate words from embeddings
    """
    def __init__(self,
                 lstm_hidden_size=50,
                 char_count=28,
                 char_embedding_size=50,):
        """
        lstm_hidden_size: hidden size of lstm, and also the embedding size of
            the words
        char_count: number of characters
        char_embedding_size: also equal to lstm_input_size
        """
        super().__init__()

        self.input_module = nn.Embedding(num_embeddings=char_count,
                                         embedding_dim=char_embedding_size)
        self.output_module = nn.Linear(lstm_hidden_size, char_count, bias=True)

        self.lstm = WrappedLSTM(char_embedding_size,
                                lstm_hidden_size,
                                input_module=self.input_module,
                                output_module=self.output_module,
                                num_layers=1)

    def forward(self, hidden, packed_input):
        return self.lstm(hidden, packed_input)

# %%
ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}


class CharDecoderHead(nn.Module):
    """
    Character level decoder LSTM to generate words from embeddings, with an
    additional fully connected layer from word embedding to hidden state.

    This module can be called with an optional use_head boolean to first pass
    the given `hidden` tensor through a fully connected layer to create the
    actual initial hidden state.

    If use_head is false, the input hidden is passed directly to the decoder.
    """

    def __init__(self,
                 lstm_hidden_size=50,
                 char_count=28,
                 char_embedding_size=50,
                 input_embedding_size=50,
                 embedding_to_hidden_activation='relu'):
        """
        lstm_hidden_size: hidden size of lstm
        char_count: number of characters
        char_embedding_size: also equal to lstm_input_size
        input_embedding_size: the size of word embeddings. A fully connected
            layer from input_embedding_size -> lstm_hidden_size is used to
            create the initial lstm hidden state from a word embedding
        """
        super().__init__()

        self.decoder = CharDecoder(lstm_hidden_size,
                                   char_count,
                                   char_embedding_size)

        self.embedding_to_hidden = nn.Sequential(
            nn.Linear(input_embedding_size, lstm_hidden_size),
            ACTS[embedding_to_hidden_activation](),
            nn.BatchNorm1d(lstm_hidden_size),
        )

    def forward(self, hidden, packed_input, use_head=True):
        """use_head=True will treat hidden as a word embedding,
        while use_head=False will pass it directly into decoder"""
        if use_head:
            h = self.embedding_to_hidden(hidden).unsqueeze(0)
            hidden = h, h
        return self.decoder(hidden, packed_input)


# Demonstrate overfit on a batch:


# from words_dataset import WordsDataset, collate_words_samples
# import torch
#
# dataset = WordsDataset('pickled_word_vecs/glove.6B.50d_words.pkl',
#             'pickled_word_vecs/glove.6B.50d_chars.pkl','cpu')
#
#
# test_model = CharDecoderHead(100, len(dataset.char2idx.keys()), 20)
#
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
#     collate_fn=collate_words_samples)
#
#
# batch = next(iter(data_loader))
# hidden = batch['embeddings'].squeeze()
# a, b = test_model(hidden, batch['packed_input'])
# # batch['packed_output'].data
# from torch import optim
# optimizer = optim.Adam(test_model.parameters())
#
# from training_functions import train_on_batch
#
# for i in range(1000):
#     train_on_batch(test_model, batch, optimizer, True)
#
# train_on_batch(test_model, batch, optimizer, True)
#
# import numpy as np
#
# # %%
# test_model.eval();
# C=np.random.randint(32)
# hid = batch['embeddings'][0,C].unsqueeze(0)
# hid.shape
# #hid = h0,h0
# t = torch.LongTensor([dataset.char2idx['START']])
# inp = torch.nn.utils.rnn.pack_sequence([t])
# out = ''
#
# pout, hid = test_model(hid,inp, use_head=True)
# next_idx = pout.data.argmax()
# inp = torch.nn.utils.rnn.pack_sequence([next_idx.unsqueeze(0)])
# out += dataset.idx2char[next_idx.item()]
# while True:
#     pout, hid = test_model(hid,inp, use_head=False)
#     next_idx = pout.data.argmax()
#     inp = torch.nn.utils.rnn.pack_sequence([next_idx.unsqueeze(0)])
#     if dataset.idx2char[next_idx.item()] == 'END':
#         break
#     out += dataset.idx2char[next_idx.item()]
# out, batch['words'][C]
#


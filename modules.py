"""Modules for building the char rnn"""

import torch
from torch import nn
import torch.functional as F

from pytorch_utils.wrapped_lstm import WrappedLSTM

class CharDecoder(nn.Module):
    """
    Character level decoder LSTM to generate words from embeddings
    """
    def __init__(self,
                 lstm_hidden_size,
                 char_count,
                 char_embedding_size,):
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


# from words_dataset import WordsDataset, collate_words_samples
#
# dataset = WordsDataset('pickled_word_vecs/glove.6B.50d_words.pkl',
#             'pickled_word_vecs/glove.6B.50d_chars.pkl','cpu')
#
#
# test_model = CharDecoder(50, len(dataset.char2idx.keys()), 20)
#
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
#     collate_fn=collate_words_samples)
#
#
# batch = next(iter(data_loader))
# hidden = (batch['embeddings'], batch['embeddings'])
# hidden[0].shape
# packed_out, hid = test_model(hidden, batch['packed_input'])
# packed_out
# criterion = nn.CrossEntropyLoss()
#
# criterion(packed_out.data, batch['packed_output'].data)

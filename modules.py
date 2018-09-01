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


# Demonstrate overfit on a batch:


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
# # hidden = batch['embeddings'], batch['embeddings']
# # a, b = test_model(hidden, batch['packed_input'])
# # batch['packed_output'].data
# from torch import optim
# optimizer = optim.Adam(test_model.parameters())
#
# from char_decoder_train import train_on_batch
#
# for i in range(1000):
#     train_on_batch(test_model, batch, optimizer)
#
# train_on_batch(test_model, batch, optimizer)
#
# C=5
# h0 = batch['embeddings'][0,C].unsqueeze(0).unsqueeze(0)
# h0.shape
# hid = h0,h0
# t = torch.LongTensor([dataset.char2idx['START']])
# inp = torch.nn.utils.rnn.pack_sequence([t])
# out = ''
# while True:
#     pout, hid = test_model(hid,inp)
#     next_idx = pout.data.argmax()
#     inp = torch.nn.utils.rnn.pack_sequence([next_idx.unsqueeze(0)])
#     if dataset.idx2char[next_idx.item()] == 'END':
#         break
#     out += dataset.idx2char[next_idx.item()]
# out, batch['words'][C]

"""
Dataset for handling words and word vectors. Reads from torch saved files which have
word vectors in a tensor and word to index maps and inverse maps.

Also requires a file which stores character level index maps.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class WordsDataset(Dataset):
    """
    Dataset of words and their word embeddings.
    Also stores indexed versions of the words using character mappings.
    """
    def __init__(self, word2vec_file, charidx_file, device='cpu'):
        """
        Requires two pickled files as inputs.

        word2vec_file when read should be a tuple with 3 elements:
        word2idx mapping (dict),
        idx2word mapping (dict),
        a torch Tensor containing word vectors.

        charidx_file when read should be a tuple with 2 elements:
        char2idx mapping (dict),
        idx2char mapping (dict).

        The char2idx dictionary must have keys called 'START' and 'END' which
        denote the start and end of character sequences(words).
        """

        self.word2idx, self.idx2word, self.embed = torch.load(word2vec_file)
        self.embed = self.embed.to(device)
        self.embed.requires_grad = False
        print(f"Loaded word2vec to {device}")

        self.char2idx, self.idx2char = torch.load(charidx_file)
        print("Loaded char2idx")


        self.indexed_words = []
        for i in range(len(self.word2idx.keys())):
            word = self.idx2word[i]
            indexed_word = torch.LongTensor(
                [self.char2idx['START']]+
                [self.char2idx[c] for c in word]+
                [self.char2idx['END']]
            ).to(device)

            indexed_word.requires_grad = False

            self.indexed_words.append(indexed_word)

        print(f"Loaded indexed words to {device}")

    def __len__(self):
        return len(self.word2idx.keys())

    def __getitem__(self, idx):
        """
        Each sample is a dict that contains 3 elements:
        word: the word as a string
        indexed_word: torch LongTensor contained indices of each char in the word
        embedding: torch Tensor which is the embedding of the word
        """
        # print(idx)
        idx = int(idx)
        sample = {'word':self.idx2word[idx],
                  'indexed_word':self.indexed_words[idx],
                  'embedding':self.embed[idx]}
        return sample



def collate_words_samples(samples):
    """
    Collates samples from a WordsDataset into a batch. To be used
    as the collate_fn for a dataloader with this dataset.

    Returns a dict with 4 elements:
    words: a list of string representations of the words
    embeddings: a torch Tensor containing each of the word embeddings,
            in the same order as the words list
    packed_input: packed_sequence which serves as input to a decoder model,
            i.e. the START token is appended to the start of the word.
    packed_output: packed_sequence which serves as target of a decoder model,
            i.e. the END token is appended to the end of the word.
    """

    # raw string representation of the words
    # must be sorted to form a packed sequence
    samples = sorted(samples, key=lambda s: -len(s['word']))
    words = [s['word'] for s in samples]

    # hidden state of shape (1, batch_size, hidden_dim).
    # the 1 corresponds to num_layers*num_directions
    embeddings = torch.stack(
        [s['embedding'] for s in samples]
    ).view(1, len(samples), -1)

    input_words = ([s['indexed_word'][:-1] for s in samples])
    output_words = ([s['indexed_word'][1:] for s in samples])

    packed_input = nn.utils.rnn.pack_sequence(input_words)
    packed_output = nn.utils.rnn.pack_sequence(output_words)

    return {'words':words,
            'embeddings':embeddings,
            'packed_input':packed_input,
            'packed_output':packed_output}

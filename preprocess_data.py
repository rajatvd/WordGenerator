"""
Helper script to preprocess glove word vector files
"""
import re
import string
import os

import torch
from sacred  import Experiment
from torchnlp.word_to_vector import GloVe, CharNGram, FastText


MAPPING = {
    'glove':GloVe,
    'char_n_gram':CharNGram,
    'fast_text':FastText
}

ex = Experiment(name="preprocess_glove_vectors")

@ex.config
def config():
    """Main config for data preprocessing
    char_n_gram currently not supported"""

    glove_file = 'glove.6B/glove.6B.300d.txt' # optional filename of glove file (.txt)
    path, file = os.path.split(glove_file)
    name, ext = os.path.splitext(file)
    output_path = f'pickled_word_vecs/{name}' # output path for pickled files


    word_vectors = 'glove' # one of 'glove', 'fast_text', 'char_n_gram'. Overrides glove_file
    word_vectors_kwargs = {
        'glove':{
            'name':'6B',
            'cache':'glove.6B/',
            'dim':300,
        },
        'fast_text':{
            'cache':'fast_text/',
            'language':'en',
            'aligned':True,
        },
        'char_n_gram':{
            'cache':'char_n_gram/',
        }
    }


    lowercase = True # whether to include lowercase letters
    uppercase = False # whether to include uppercase letters
    chars = '' # characters to add to lowercase, uppercase if included

    letters = chars # final set of letters used to filter words

    letters += (string.ascii_lowercase if lowercase else '')
    letters += (string.ascii_uppercase if uppercase else '')

@ex.capture
def load_glove_model(glove_file, _log):
    """
    Loads word vectors from a glove txt file
    return a dictionary mapping words to word vectors(torch tensors)
    """
    with open(glove_file, 'r', encoding='utf-8') as file:
        _log.info(f"Opened {glove_file}")
        model = {}
        for line in file:
            split_line = line.split()
            word = split_line[0]
            embedding = torch.FloatTensor(
                [float(val) for val in split_line[1:]]
            )
            embedding.requires_grad = False
            model[word] = embedding
    #    print ("Done.",len(model)," words loaded!")
        return model

# word_vectors_dict =  load_glove_model('glove.6B/glove.6B.50d.txt')


def remove_duplicate_chars(letters):
    """Remove duplicate characters in the letters argument and return a string
    with the remaining characters sorted."""
    return ''.join(sorted(list(set(list(letters)))))

# letters = "abcabcdeffffghu"
# letters = remove_duplicate_chars(letters=letters)

def filter_words(word_vectors_dict, letters):
    """
    Remove words from the word_vectors_dict which contain characters not
    present in the letters argument.
    """
    out = {}
    for word, vec in word_vectors_dict.items():
        if re.match(r'^['+letters+r']+$', word):
            out[word] = vec

    return out

# filted = filter_words(w2v, letters)
# len(filted)

@ex.capture
def pickle_word_vecs(output_path, word_vectors_dict, letters, _log):
    """
    Filters the given word_vectors_dict to have words made only using the given
    letters.
    Saves two files with names <output_path>_words.pkl and <output_path>_chars.pkl
    which contain the following:

    <output_path>_words.pkl: a tuple of 3 elements
        word2idx mapping (dict),
        idx2word mapping (dict),
        a torch Tensor containing word vectors.

    <output_path>_chars.pkl: a tuple of 2 elements
        char2idx mapping (dict),
        idx2char mapping (dict).
    """
    # clean up letters
    letters = remove_duplicate_chars(letters)

    _log.info(f"Filtering words with following letters : {letters}")

    # filter w2v
    filtered_w2v = filter_words(word_vectors_dict, letters)

    # get words in sorted order
    words = sorted(list(filtered_w2v.keys()))


    word_count = len(words)
    embedding_dim = len(word_vectors_dict[words[0]])

    _log.info(f"{word_count} words with embedding dimension of {embedding_dim}")

    word2idx = {}
    idx2word = {}
    word_vecs = torch.zeros((word_count, embedding_dim), requires_grad=False)
    i = 0
    for word in words:
        word2idx[word] = i
        idx2word[i] = word
        word_vecs[i] = filtered_w2v[word]
        i += 1


    # make directory and save pickle files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    word_vec_file = f"{output_path}_words.pkl"
    torch.save((word2idx, idx2word, word_vecs), word_vec_file)

    _log.info(f"Saved word vecs and mappings to {word_vec_file}")

    chars = list(letters) + ['START', 'END']
    char2idx = dict(zip(chars, range(len(chars))))
    idx2char = dict(zip(range(len(chars)), chars))
    char_file = f"{output_path}_chars.pkl"

    torch.save((char2idx, idx2char), char_file)

    _log.info(f"Saved char mappings to {char_file}")

@ex.automain
def main(glove_file, word_vectors, output_path, letters, _config):
    """Main function to preprocess data"""
    # print(_config)
    if word_vectors not in MAPPING.keys():
        word_vectors_dict = load_glove_model(glove_file)
    else:
        w2v = MAPPING[word_vectors](**_config['word_vectors_kwargs'][word_vectors])
        word_vectors_dict = {word:w2v.vectors[index] for word, index in w2v.stoi.items()}

    pickle_word_vecs(output_path, word_vectors_dict, letters)

"""Ingredient for making a model with a wrapped lstm and a
head for hidden state, using the CharDecoderHead."""

import torch

from sacred import Ingredient
from words_dataset import WordsDataset, collate_words_samples

data_ingredient = Ingredient('dataset')

@data_ingredient.config
def data_config():
    """Config for data source and loading"""
    batch_size = 2
    word2vec_file = 'pickled_word_vecs/glove.6B.300d_words.pkl'
    charidx_file = 'pickled_word_vecs/glove.6B.300d_chars.pkl'
    device = 'cpu'
    val_split = 0.01
    num_workers = 0 # number of subprocesses apart from main for data loading

@data_ingredient.capture
def make_dataloaders(word2vec_file, charidx_file,
                     batch_size, num_workers, val_split, device, _log):
    """Make the dataloader using the given paths to pickled files"""
    dset = WordsDataset(word2vec_file, charidx_file, device)

    _log.info("Loaded dataset")

    total = len(dset)
    train_num = int(total*(1-val_split))
    val_num = total-train_num

    _log.info(f"Split dataset into {train_num} train samples and {val_num} \
    validation samples")

    train, val = torch.utils.data.dataset.random_split(dset,
                                                       [train_num, val_num])

    # pin = device!='cpu'
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # pin_memory=pin,
        collate_fn=collate_words_samples,)

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # pin_memory=pin,
        collate_fn=collate_words_samples,)


    return train_loader, val_loader

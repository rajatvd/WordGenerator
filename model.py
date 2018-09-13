"""Ingredient for making a model with a wrapped lstm and a
head for hidden state, using the CharDecoderHead."""

import torch

from sacred import Ingredient
from modules import CharDecoderHeadRNN

model_ingredient = Ingredient('model')

@model_ingredient.config
def model_config():
    """Config for model"""
    mode = 'GRU'
    dropout = 0.5
    hidden_size = 500
    char_count = 28
    char_embedding_size = 300
    word_embedding_size = 300
    embedding_to_hidden_activation = 'relu' # relu, tanh, sigmoid
    device='cpu'

@model_ingredient.capture
def make_model(mode,
               hidden_size,
               char_count,
               char_embedding_size,
               word_embedding_size,
               embedding_to_hidden_activation,
               device,
               dropout,
               _log):
    """Create char decoder model from config"""
    model = CharDecoderHeadRNN(mode,
                               hidden_size,
                               char_count,
                               char_embedding_size,
                               input_embedding_size=word_embedding_size,
                               dropout=dropout,
                               embedding_to_hidden_activation=embedding_to_hidden_activation).to(device)

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    num_params = len(params)
    _log.info(f"Created char decoder head {mode} model with {num_params} parameters \
    on {device}")
    return model

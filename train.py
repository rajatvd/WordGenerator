"""
Run this script to train a character level decoder LSTM on the glove dataset
"""
from functools import partial

import torch
from torch import optim

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from visdom_observer.visdom_observer import VisdomObserver
import pytorch_utils.sacred_trainer as st
from pytorch_utils.updaters import averager


from modules import CharDecoder, CharDecoderHead
from training_functions import train_on_batch, create_scheduler_callback
from words_dataset import collate_words_samples, WordsDataset

torch.backends.cudnn.benchmark = True

SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('characterlevel_decoder')
SAVE_DIR = 'CharDecoderLSTM'
ex.observers.append(FileStorageObserver.create(SAVE_DIR))
ex.observers.append(VisdomObserver())

# -------------DATA--------------

@ex.config
def data_config():
    """Config for data source and loading"""
    batch_size = 32
    word2vec_file = 'pickled_word_vecs/glove.6B.300d_words.pkl'
    charidx_file = 'pickled_word_vecs/glove.6B.300d_chars.pkl'
    device = 'cpu'
    num_workers = 0 # number of subprocesses apart from main for data loading

@ex.capture
def make_dataloader(word2vec_file, charidx_file,
                    batch_size, num_workers, device):
    """Make the dataloader using the given paths to pickled files"""
    dset = WordsDataset(word2vec_file, charidx_file, device)
    # pin = device!='cpu'
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # pin_memory=pin,
        collate_fn=collate_words_samples,
    )

    return loader


# ----------------OPTIMIZER-----------------

@ex.config
def optimizer_config():
    """Config for optimzier
    Currently available opts (types of optimizers):
        adam
        adamax
        rmsprop
    """
    lr = 0.001 # learning rate
    opt = 'adam' # type of optimzier
    weight_decay = 0 # l2 regularization weight_decay (lambda)


@ex.capture
def make_optimizer(model, lr, opt, weight_decay):
    """Make an optimizer of the given type (opt), for the given model's
    parameters with the given learning rate (lr)"""
    optimizers = {
        'adam':optim.Adam,
        'adamax':optim.Adamax,
        'rmsprop':optim.RMSprop,
    }

    optimizer = optimizers[opt](model.parameters(), lr=lr,
                                weight_decay=weight_decay)

    return optimizer


# -----------CALLBACK FOR LR SCHEDULING-------------

@ex.config
def scheduler_config():
    """Config for lr scheduler"""
    milestones = [50, 100]
    gamma = 0.5

@ex.capture
def make_scheduler_callback(optimizer, milestones, gamma):
    """Create a MultiStepLR scheduler callback for the optimizer
    using the config"""
    return create_scheduler_callback(optimizer, milestones, gamma)


# -----------MODEL-------------

@ex.config
def model_config():
    """Config for model"""
    lstm_hidden_size = 500
    char_count = 28
    char_embedding_size = 300
    word_embedding_size = 300
    embedding_to_hidden_activation = 'relu' # relu, tanh, sigmoid

@ex.capture
def make_model(lstm_hidden_size,
               char_count,
               char_embedding_size,
               word_embedding_size,
               embedding_to_hidden_activation,
               device,
               _log):
    """Create char decoder model from config"""
    model = CharDecoderHead(lstm_hidden_size,
      char_count,
      char_embedding_size,
      input_embedding_size=word_embedding_size,
      embedding_to_hidden_activation=embedding_to_hidden_activation).to(device)

    params = torch.nn.utils.parameters_to_vector(model.parameters())
    num_params = len(params)
    _log.info(f"Created model with {num_params} parameters")
    return model



@ex.config
def train_config():
    epochs = 100
    save_every = 1
    start_epoch = 1


@ex.automain
def main(_run):

    loader = make_dataloader()
    model = make_model()
    optimizer = make_optimizer(model)
    callback = make_scheduler_callback(optimizer)

    st.loop(
        _run=_run,
        model=model,
        optimizer=optimizer,
        save_dir=SAVE_DIR,
        trainOnBatch=partial(train_on_batch, use_head=True),
        train_loader=loader,
        callback=callback,
        callback_metric_names=['learning_rate'],
        **_run.config,
        batch_metric_names=['loss'],
        updaters=[averager])




# from torch import nn
#
# def sample(rnn,sample,char2idx,idx2char, random_embed=False, sigma=0.1):
#     with torch.no_grad():
#         word = sample['word']
#         indexed_word = sample['indexed_word']
#         embedding = sample['embedding'].view(1,-1)
#
#         # print(embedding.sum())
#         hidden = embedding + sigma*torch.randn_like(embedding)
#         # print(rnn.training)
#         if random_embed:
#             hidden = sigma*torch.randn_like(embedding)
#
#         idx = -1
#         inp = [torch.LongTensor([char2idx['START']])]
#         inp = nn.utils.rnn.pack_sequence(inp)
#         word_out=''
#
#         out,hidden = rnn(hidden, inp, use_head=True)
#
#         pred = out.data.cpu().numpy().reshape(-1)
#         #print(pred.shape)
#         idx = np.argmax(pred)
#         word_out += idx2char[idx]
#         inp = [torch.LongTensor([idx])]
#         inp = nn.utils.rnn.pack_sequence(inp)
#
#         max_len=30
#         c=0
#         while idx != char2idx['END']:
#             out,hidden = rnn(hidden, inp, use_head=False)
#
#
#             pred = out.data.detach().cpu().numpy().reshape(-1)
#             #print(pred.shape)
#             idx = np.argmax(pred)
#             if idx == char2idx['END'] or c>max_len:
#                 return word_out
#             word_out += idx2char[idx]
#             inp = [torch.LongTensor([idx])]
#             inp = nn.utils.rnn.pack_sequence(inp)
#             c+=1
#
#
#
#
# # %%
#
# # model_file = 'Z:\\UbuntuVMShared\\Notebooks\\CharRNN\\CharDec'+
# # 'oderLSTM\\20\\epoch578_02-09_0917_learning_rate0.0001_loss1.6675.statedict.pkl'
# # model_file = 'CharDecoderLSTM\\23\\epoch524_02-09_2230_learning_rate0.0001_loss1.0834.statedict.pkl'
# model_file = 'CharDecoderLSTM\\35\\epoch481_07-09_1407_learning_rate0.0000_loss0.0117.statedict.pkl'
#
# word2vec_file = 'pickled_word_vecs/glove.6B.300d_words.pkl'
# charidx_file = 'pickled_word_vecs/glove.6B.300d_chars.pkl'
# device = 'cpu'
#
# # %%
# model = CharDecoderHead(1024, 28, 300, 300).to('cpu')
# model.load_state_dict(torch.load(model_file))
# model = model.eval();
# import numpy as np
# dataset = WordsDataset(word2vec_file, charidx_file, device)
#
# # %%
# inds = np.random.choice(len(dataset), 30)
# for i in inds:
#     print(dataset[i]['word'],end='\t\t')
#     print(sample(model,dataset[i],dataset.char2idx,dataset.idx2char, sigma=0))
# # %%
# word = 'constitution'
# print(word+":")
# for i in range(10):
#     print(sample(model,dataset[dataset.word2idx[word]],dataset.char2idx,dataset.idx2char,
#     random_embed=True, sigma=1))














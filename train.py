"""
Run this script to train a character level decoder LSTM on the glove dataset
"""

import torch
from torch import optim

from sacred import Experiment
from sacred.observers import FileStorageObserver
from visdom_observer.visdom_observer import VisdomObserver
import pytorch_utils.sacred_trainer as st
from pytorch_utils.updaters import averager


from modules import CharDecoder
from training_functions import train_on_batch, create_scheduler_callback
from words_dataset import collate_words_samples, WordsDataset

torch.backends.cudnn.benchmark = True

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


@ex.capture
def make_optimizer(model, lr, opt):
    """Make an optimizer of the given type (opt), for the given model's
    parameters with the given learning rate (lr)"""
    optimizers = {
        'adam':optim.Adam,
        'adamax':optim.Adamax,
        'rmsprop':optim.RMSprop,
    }

    optimizer = optimizers[opt](model.parameters(), lr=lr)

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
    lstm_hidden_size = 300
    char_count = 28
    char_embedding_size = 300

@ex.capture
def make_model(lstm_hidden_size, char_count, char_embedding_size, device, _log):
    """Create char decoder model from config"""
    model = CharDecoder(lstm_hidden_size, char_count,
        char_embedding_size).to(device)

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
        trainOnBatch=train_on_batch,
        train_loader=loader,
        callback=callback,
        callback_metric_names=['learning_rate'],
        **_run.config,
        batch_metric_names=['loss'],
        updaters=[averager])




# from torch import nn
#
# def sample(rnn,sample,char2idx,idx2char):
#     with torch.no_grad():
#         word = sample['word']
#         indexed_word = sample['indexed_word']
#         embedding = sample['embedding'].view(1,1,-1)
#
#         hidden = embedding,embedding
#
#         idx = -1
#         inp = [torch.LongTensor([char2idx['START']])]
#         inp = nn.utils.rnn.pack_sequence(inp)
#         word_out=''
#         while idx != char2idx['END']:
#             out,hidden = rnn(hidden, inp)
#
#
#             pred = out.data.detach().cpu().numpy().reshape(-1)
#             #print(pred.shape)
#             idx = np.argmax(pred)
#             if idx == char2idx['END']:
#                 return word_out
#             word_out += idx2char[idx]
#             inp = [torch.LongTensor([idx])]
#             inp = nn.utils.rnn.pack_sequence(inp)
#
#
#
#
#
#
# word2vec_file = 'pickled_word_vecs/glove.6B.300d_words.pkl'
# charidx_file = 'pickled_word_vecs/glove.6B.300d_chars.pkl'
# device = 'cpu'
#
#
# model = CharDecoder(300, 28,300).to('cpu')
# model.load_state_dict(torch.load('Z:\\UbuntuVMShared\\Notebooks\\CharRNN\\CharDec'+
# 'oderLSTM\\20\\epoch578_02-09_0917_learning_rate0.0001_loss1.6675.statedict.pkl'))
#
# import numpy as np
# dataset = WordsDataset(word2vec_file, charidx_file, device)
#
# inds = np.random.choice(len(dataset), 30)
# for i in inds:
#     print(dataset[i]['word'],end='\t\t')
#     print(sample(model,dataset[i],dataset.char2idx,dataset.idx2char))
#
# word = 'agility'
# print(dataset[dataset.word2idx[word]]['word'])
# print(sample(model,dataset[dataset.word2idx[word]],dataset.char2idx,dataset.idx2char))

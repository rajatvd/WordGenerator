"""Run beam search sampling using a given rundir"""
# %%
import os
import json
from sacred import Experiment
import torch
from beam_search import sample_beam, pass_word

from model import make_model
from dataset import make_dataloaders

# %%

def read_config(run_dir):
    """Read the config json from the given run directory"""
    with open(os.path.join(run_dir, 'config.json')) as file:
        config = json.loads(file.read())
        del config['model']['__doc__']
        del config['dataset']['__doc__']

    return config

def get_model_path(run_dir, epoch):
    """Get the path to the saved model state_dict with the given epoch number
    If epoch is 'latest', the latest model state dict path will be returned.
    """
    if epoch == 'latest':
        return os.path.join(run_dir, 'latest.statedict.pkl')

    filenames = os.listdir(run_dir)

    for filename in filenames:
        if 'statedict' not in filename:
            continue
        if filename.startswith('epoch'):
            number = int(filename[len('epoch'):].split('_')[0])
            if epoch == number:
                return os.path.join(run_dir, filename)

    raise ValueError(f"No statedict found with epoch number '{epoch}'")

def get_embedding(config, word, sigma, device='cpu', dset=None):
    """Get the word embedding to be used for inference"""
    with torch.no_grad():
        perturbation = sigma*torch.randn(config['model']['word_embedding_size'])
        perturbation = perturbation.to(device)
        if word == 0:
            return perturbation

        embedding = dset[dset.word2idx[word]]['embedding']
        return embedding + perturbation

def sample(model, config, input, sigma, dset, char2idx, idx2char,
           beam_size, max_len, device, start='START',
           use_head=True):
    """Perform sampling using the given model and config.
    input can be one of three things:
     - a word: the embedding of the word + noise will be used
     - an embedding: the embedding + noise will be used
     - the integer 0: a random embedding with given sigma will be used

    Returns
     - beam of sampled words in decreasing order of probabilities
     - a list of probabilities for these words
     - the word in the vocabulary which is closest to the input embedding after
        adding noise
    """

    if type(input)==int or type(input)==str:
        input_embedding = get_embedding(config, input, sigma, device, dset)
    else:
        input_embedding = input
        perturbation = sigma*torch.randn(config['model']['word_embedding_size'])
        perturbation = perturbation.to(device)
        input_embedding = input_embedding + perturbation

    # find closest glove word to the embedding
    cos_sim = torch.nn.modules.distance.CosineSimilarity()
    index = int(cos_sim(input_embedding.unsqueeze(0), dset.embed).argmax())
    closest = dset.idx2word[index]

    use_head = False
    if start != 'START':
        out, input_embedding = pass_word(start[:-1], model, input_embedding, char2idx, device)
        start_c = start[-1]
    else:
        use_head = True
        start_c = 'START'

    samples, probabs = sample_beam(model, input_embedding, char2idx, idx2char,
                                   use_head=use_head,
                                   k=beam_size, maxlen=max_len, start=start_c)

    if start != 'START':
        samples = [start+s for s in samples]

    return samples, probabs, closest

def probab_word(word, model, input_embedding, char2idx, device, use_head=True):
    """Pass a word through the given model using the input_embedding, and find
    the probability of that word being generated."""
    out, hidden = pass_word(word, model, input_embedding, char2idx, device, use_head=True)
    out = torch.softmax(out.data, dim=1)
    expected = [char2idx[c] for c in word]+[char2idx['END']]
    probabs = out.data[range(len(expected)),expected]
    return probabs.prod()

# %%
ex = Experiment('sampling')

@ex.config
def input_config():
    """Parameters for sampling using the given model"""
    run_dir = 'trained_model'
    epoch = 'latest'
    beam_size = 20 # currently should be < 28
    sigma = 0.3 # sigma of gaussian noise to add to embedding
    word = 0 # input word embedding to use. if equal to integer 0, a random embedding will be used
    max_len = 30 # maximum length of a sampled word
    num_samples = 10 # number of times to sample
    print_probabs = False # whether to print beam search probabilities
    start = 'START' # all sampled words will be forced to start with this, use 'START' for no restriction on starting
    device = 'cpu'

@ex.automain
def main(run_dir, epoch, beam_size, max_len, word, sigma,
         device, num_samples, print_probabs, _log, start):

    config = read_config(run_dir)
    _log.info(f"Read config from {run_dir}")

    model = make_model(**{**config['model'], 'device':device}, _log=_log)
    path = get_model_path(run_dir, epoch)
    model.load_state_dict(torch.load(path))
    model = model.eval()
    _log.info(f"Loaded state dict from {path}")

    dset, train_loader, val_loader = make_dataloaders(**{**config['dataset'],
                                                         'device':device},
                                                      _log=_log)
    char2idx, idx2char = torch.load(config['dataset']['charidx_file'])

    print("sigma = 0:")
    samples, probabs, closest = sample(model, config, word, 0, dset, char2idx, idx2char,
                                       beam_size, max_len, device, start=start)
    print(f"Closest word: {closest}")
    if print_probabs:
        print(f"Word\tProbability")
        for word_sample, probab in zip(samples, probabs):
            print(f"{word_sample}\t{probab}")
    else:
        print(" ".join(samples))

    print(f"Word: {word}, sigma={sigma}")
    for i in range(num_samples):
        samples, probabs, closest = sample(model, config, word, sigma, dset, char2idx, idx2char,
                                           beam_size, max_len, device, start=start)

        print(f"Closest word: {closest}")
        if print_probabs:
            print(f"Word\tProbability")
            for word_sample, probab in zip(samples, probabs):
                print(f"{word_sample}\t{probab}")
        else:
            print(" ".join(samples))

        print()

# Run using hydrogen or jupyter for easy sampling

# # %%
# from IPython.display import display, Markdown
# run_dir = 'trained_model'
# epoch = 'latest'
# beam_size = 10 # currently should be < 28
# sigma = 1 # sigma of gaussian noise to add to embedding
# word = 0 # input word embedding to use. if equal to integer 0, a random embedding will be used
# max_len = 30 # maximum length of a sampled word
# num_samples = 10 # number of times to sample
# print_probabs = False # whether to print beam search probabilities
# device = 'cpu'
#
# import logging
# log = logging.getLogger('sampling')
#
# # %%
# config = read_config(run_dir)
# model = make_model(**{**config['model'], 'device':device}, _log=log)
# path = get_model_path(run_dir, epoch)
# model.load_state_dict(torch.load(path))
# model = model.eval()
#
# # %%
# dset, train_loader, val_loader = make_dataloaders(**{**config['dataset'],
# 'device':device}, _log=log)
# char2idx, idx2char = torch.load(config['dataset']['charidx_file'])
#
# # %%
# word = 'research'
# sigma = 0.2
# beam_size = 20
# start = 'START'
#
# # # %%
# # words = ['fleeting']
# # embeds = torch.stack([dset.embed[dset.word2idx[w]] for w in words])
# # word = embeds.mean(dim=0)
#
# # %%
# out = f"Words: {word}, sigma={sigma}, start={start}\n\n"
# out += "sigma = 0 "
# samples, probabs, closest = sample(model, config, word, 0, dset, char2idx, idx2char,
#                           beam_size, max_len, device, start=start)
# out += f"Closest word: {closest}  \n"
# out += " ".join(samples) + "\n\n"
# for i in range(num_samples):
#     samples, probabs, closest = sample(model, config, word, sigma, dset, char2idx, idx2char,
#                               beam_size, max_len, device, start=start)
#     out += f"Closest word: {closest}  \n"
#     out += " ".join(samples) + "\n\n"
#
# w = 'temple'
# p = probab_word(w, model, dset.embed[dset.word2idx[word]], dset.char2idx, device)
# # out += f"Probability of {w} = {p}"
# display(Markdown(out))

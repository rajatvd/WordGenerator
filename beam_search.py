"""Perform beam search on a decoder rnn with head layer"""

import torch
from torch import nn

import torch.nn.utils.rnn as p
pack = p.pack_sequence

def sample_beam(model, input_embedding, char2idx, idx2char, k=5, maxlen=30):
    """Sample using beam search
    model: model to be used. It must have a head layer (or a use_head option in
        forward)
    input_embedding: The input embedding
    char2idx: dict which maps characters to one hot indices
        Must have 'START' and 'END' as keys
    idx2char: dict which maps one hot indices to characters
    k: size of the beam
    maxlen: maximum length of a sampled word
    """
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        input_embedding = input_embedding.view(1, -1)
        device = input_embedding.device

        inp = [torch.LongTensor([char2idx['START']]).to(device)]
        inp = nn.utils.rnn.pack_sequence(inp)
        out, hidden = model(input_embedding, inp, use_head=True)

        out = softmax(out.data).view(-1).cpu().numpy()
        max_k = argsort(out)[-k:][::-1]
        oldprobs = out[max_k]
        words = [[i] for i in max_k]
        inp = pack([torch.LongTensor([j]).to(device) for j in max_k])

        hidden0 = torch.cat([hidden[0] for i in range(k)], dim=1)
        hidden1 = torch.cat([hidden[1] for i in range(k)], dim=1)

        hidden = hidden0, hidden1
        WORDS = []
        for c in range(maxlen):
            out, hidden = model(hidden, inp, use_head=False)
            out = softmax(out.data).cpu().numpy()

            #print(out.shape)
            inpnp = inp.data.detach().cpu().numpy()
            done = where(inpnp == char2idx['END'])
            out[done] = 0
            if len(out[done]) != 0:
                #print(out[done].shape)
                for d in done[0]:
                    out[d][char2idx['END']] = 1
                #print(done)

            #print(out)
            #print(out[done])
            out = (oldprobs.reshape(-1, 1)*out)
            max_k = argsort(out)[:, -k:][:, ::-1]

            #print(max_k)
            probs = array([out[i][max_k[i]] for i in range(k)])
            #print(probs)
            flat = probs.reshape(-1)
            max_k2 = argsort(flat)[::-1][:k]
            word_inds = max_k2//k
            next_chars_inds = max_k2%k

            oldprobs = flat[max_k2]
            #print(oldprobs)

            new_words = []
            new_inp = []
            for i, word_ind in enumerate(word_inds):
                next_char = max_k[word_ind][next_chars_inds[i]]
                if next_char == char2idx['END']:
                    #print("HIT AN END at word {}".format(word_ind))
                    WORDS.append((words[word_ind], oldprobs[i]))
                    #the_word = words[word_ind]
                    #return ''.join([idx2char[i] for i in the_word])
                new_inp.append(torch.LongTensor([next_char]).cuda())
                word = words[word_ind][:]
                word = word + [next_char]
                new_words.append(word)
            words = new_words[:]


            h1, h2 = hidden
            h1, h2 = h1[0][word_inds].view(1, k, -1), h2[0][word_inds].view(1, k, -1)
            hidden = h1, h2

            inp = pack(new_inp)

        return [''.join([idx2char[i] for i in word if i != char2idx['END']]) for word in words], oldprobs

# %%
# import json
# rundir = 'CharDecoderLSTM\\24'
# with open(f'{rundir}\\config.json') as f:
#     cfg = json.loads(f.read())
#
# cfg
#
# from model_lstm import make_model, model_ingredient
# from numpy import *
# def f():
#     pass
#
# f.info = lambda x:None
#
# model = make_model(cfg['lstm_hidden_size'], cfg['char_count'],  cfg['char_embedding_size'], cfg['word_embedding_size'],cfg['embedding_to_hidden_activation'], 'cuda', f)
#
# model = model.eval()
# model.load_state_dict(torch.load(f'{rundir}\\epoch595_03-09_0955_learning_rate0.0000_loss1.0509.statedict.pkl'))
# char2idx, idx2char = torch.load(cfg['charidx_file'])
#
# sample_beam(model, 1*torch.randn(300).cuda(), char2idx, idx2char, k=20)

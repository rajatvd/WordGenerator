"""Perform beam search on a decoder rnn with head layer"""

import torch
from torch import nn
import numpy as np

import torch.nn.utils.rnn as p
pack = p.pack_sequence

def sample_beam(model, input_embedding, char2idx, idx2char, k=5, maxlen=30,
                start='START', use_head=True):
    """Sample using beam search
    model: model to be used. It must have a head layer (or a use_head option in
        forward)
    input_embedding: The input embedding
    char2idx: dict which maps characters to one hot indices
        Must have 'START' and 'END' as keys
    idx2char: dict which maps one hot indices to characters
    k: size of the beam
    maxlen: maximum length of a sampled word
    start: which character to start with
    use_head: whether to pass the input_embedding through the head layer for the
        first beam expansion
    """
    with torch.no_grad():
        device = input_embedding.device
        softmax = nn.Softmax(dim=1)
        if use_head:
            input_embedding = input_embedding.view(1, -1)

        inp = [torch.LongTensor([char2idx[start]]).to(device)]
        inp = nn.utils.rnn.pack_sequence(inp)
        out, hidden = model(input_embedding, inp, use_head=use_head)

        out = softmax(out.data).view(-1).cpu().numpy()
        max_k = np.argsort(out)[-k:][::-1]
        oldprobs = out[max_k]
        words = [[i] for i in max_k]
        inp = pack([torch.LongTensor([j]).to(device) for j in max_k])

        if model.mode == 'LSTM':
            hidden0 = torch.cat([hidden[0] for i in range(k)], dim=1)
            hidden1 = torch.cat([hidden[1] for i in range(k)], dim=1)
            hidden = hidden0, hidden1
        else:
            hidden = torch.cat([hidden for i in range(k)], dim=1)
        WORDS = []
        for c in range(maxlen):
            out, hidden = model(hidden, inp, use_head=False)
            out = softmax(out.data).cpu().numpy()

            #print(out.shape)
            inpnp = inp.data.detach().cpu().numpy()
            done = np.where(inpnp == char2idx['END'])
            out[done] = 0
            if len(out[done]) != 0:
                #print(out[done].shape)
                for d in done[0]:
                    out[d][char2idx['END']] = 1
                #print(done)

            #print(out)
            #print(out[done])
            out = (oldprobs.reshape(-1, 1)*out)
            max_k = np.argsort(out)[:, -k:][:, ::-1]

            #print(max_k)
            probs = np.array([out[i][max_k[i]] for i in range(k)])
            #print(probs)
            flat = probs.reshape(-1)
            max_k2 = np.argsort(flat)[::-1][:k]
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
                new_inp.append(torch.LongTensor([next_char]).to(device))
                word = words[word_ind][:]
                word = word + [next_char]
                new_words.append(word)
            words = new_words[:]

            if model.mode == 'LSTM':
                h1, h2 = hidden
                h1, h2 = h1[0][word_inds].view(1, k, -1), h2[0][word_inds].view(1, k, -1)
                hidden = h1, h2
            else:
                hidden = hidden[0][word_inds].view(1, k, -1)


            inp = pack(new_inp)

        return [''.join([idx2char[i] for i in word if i != char2idx['END']]) for word in words], oldprobs


def pass_word(word, model, input_embedding, char2idx, device, use_head=True):
    """Pass a word through the given model using the input_embedding,
    Returns the output and final hidden state"""
    inp = torch.LongTensor([char2idx['START']] + [char2idx[c] for c in word]).to(device)
    inp = pack([inp])
    out, hidden = model(input_embedding.unsqueeze(0), inp, use_head=use_head)
    return out, hidden
    
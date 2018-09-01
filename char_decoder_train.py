"""
Train on batch function for training a character level decoder lstm
"""
from torch import nn

def train_on_batch(model, batch, optimizer):
    """
    Perform one train step on the CharDecoder model using the given optimizer
    and batch of data.

    Metrics returned:
    loss
    """

    # hidden state initialized as embedding
    embeddings = batch['embeddings']
    hidden = embeddings, embeddings

    # input and target batches of sequences
    packed_input = batch['packed_input']
    packed_target = batch['packed_output']

    criterion = nn.CrossEntropyLoss()

    # forward pass
    packed_output, hidden = model(hidden, packed_input)
    loss = criterion(packed_output.data, packed_target.data)

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu().detach().numpy(),

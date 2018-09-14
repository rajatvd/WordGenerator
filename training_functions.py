"""
Train on batch function for training a character level decoder lstm
"""
import torch
from torch import nn
from torch import optim

def train_on_batch(model, batch, optimizer, use_head=False, sigma=0):
    """
    Perform one train step on the CharDecoder model using the given optimizer
    and batch of data.

    Adds `sigma` gaussian noise to the embedding before passing to model

    Metrics returned:
    loss
    """

    # hidden state initialized as embedding
    hidden = batch['embeddings']

    hidden = hidden + sigma*torch.randn_like(hidden)

    # if not using head layer, directly pass embedding as hidden state
    if use_head:
        hidden = hidden.squeeze()
    else:
        hidden = hidden, hidden

    # input and target batches of sequences
    packed_input = batch['packed_input']
    packed_target = batch['packed_output']

    criterion = nn.CrossEntropyLoss()

    # forward pass
    packed_output, hidden = model(hidden, packed_input, use_head=use_head)
    loss = criterion(packed_output.data, packed_target.data)

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu().detach().numpy(),


def validate(model, val_loader):
    """Compute loss of the model on the given `val_loader`.
    (loss with teacher forcing)
    Also assumes the model uses a head layer from embedding to hidden state
    """
    with torch.no_grad():
        model = model.eval()

        val_loss = 0
        total = 0
        for batch in val_loader:
            # embedding is input to head layer
            embedding = batch['embeddings'].squeeze()

            # input and target batches of sequences
            packed_input = batch['packed_input']
            packed_target = batch['packed_output']

            criterion = nn.CrossEntropyLoss()

            # forward pass
            packed_output, hidden = model(embedding, packed_input, use_head=True)
            loss = criterion(packed_output.data, packed_target.data)
            val_loss += loss*len(batch['words'])
            total += len(batch['words'])

        val_loss /= total
        model = model.train()
        return val_loss.cpu().numpy()

def scheduler_generator(optimizer, milestones, gamma):
    """A generator which performs lr scheduling on the given optimizer using
    a MultiStepLR scheduler with given milestones and gamma."""
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones,
                                               gamma)
    while True:
        scheduler.step()
        yield (optimizer.param_groups[0]['lr'],)

def create_scheduler_callback(optimizer, milestones, gamma):
    """Returns a function which can be used as callback for lr scheduling
    on the given optimizer using a MultiStepLR scheduler with given
    milestones and gamma."""

    g = scheduler_generator(optimizer, milestones, gamma)
    def scheduler_callback(model, val_loader, batch_metrics_dict):
        """LR scheduler callback using the next function of a
        scheduler_generator"""

        return next(g)

    return scheduler_callback

def create_val_scheduler_callback(optimizer, milestones, gamma):
    """Returns a function which can be used as callback for lr scheduling
    on the given optimizer using a MultiStepLR scheduler with given
    milestones and gamma.

    It also computes loss on the validation data loader.
    """

    g = scheduler_generator(optimizer, milestones, gamma)
    def scheduler_callback(model, val_loader, batch_metrics_dict):
        """LR scheduler callback using the next function of a
        scheduler_generator"""

        val_loss = validate(model, val_loader)
        lr = next(g)

        return val_loss, lr[0]

    return scheduler_callback

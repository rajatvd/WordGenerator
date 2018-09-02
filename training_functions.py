"""
Train on batch function for training a character level decoder lstm
"""
from torch import nn
from torch import optim

def train_on_batch(model, batch, optimizer, use_head=False):
    """
    Perform one train step on the CharDecoder model using the given optimizer
    and batch of data.

    Metrics returned:
    loss
    """

    # hidden state initialized as embedding
    hidden = batch['embeddings']

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

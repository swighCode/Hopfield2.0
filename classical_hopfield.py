import torch
import torch.nn as nn

class ClassicalHopfield(nn.Module):
    def __init__(self, X):
        super().__init__()
        if X.dim() == 1:
            X = X.view(-1, 1)
        self.num_neurons = X.size(0)
        # Compute the weight matrix using outer products and zero the diagonal
        W = torch.mm(X, X.t()) / self.num_neurons
        W.fill_diagonal_(0)
        self.register_buffer("W", W)

    def forward(self, state):
        # Asynchronously update each neuron in random order
        new_state = state.clone()
        indices = torch.randperm(self.num_neurons)
        for i in indices:
            activation = torch.dot(self.W[i], new_state)
            new_state[i] = 1.0 if activation >= 0 else -1.0
        return new_state
import torch
import torch.nn as nn

class Hopfield(nn.Module):
    def __init__(self, X, B=1.0):
        super().__init__()
        if X.dim() == 1:
            X = X.view(-1, 1)
        self.register_buffer("X", X)  # (inp_size, num_patterns)
        self.B = B  # β as a learnable parameter if desired

    def energy(self, state):
        inner_products = torch.matmul(self.X.T, state)  # (num_patterns,)
        lse = -(1/self.B) * torch.logsumexp(self.B * inner_products, dim=0)
        quad = 0.5 * torch.dot(state, state)
        E = lse + quad
        return E

    def forward(self, state):
        inner_products = torch.matmul(self.X.T, state)  # (num_patterns,)
        new_state = self.X @ torch.softmax(self.B * inner_products, dim=0)
        return new_state
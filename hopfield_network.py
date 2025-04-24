import torch
import torch.nn as nn

inp_size = 64*64

N = torch.tensor(1)
X = torch.randn(inp_size, N)
B = .5

M = torch.max(X).item()
state = torch.randn(inp_size)

lse = -torch.logsumexp(torch.matmul(X.T, state), 0) # return a float
state_mul = torch.dot(state, state)*0.5 # return a float
divers = torch.log(N) + M**2 * 0.5
E = lse + state_mul + divers
new_state = X @ torch.softmax(torch.matmul(X.T, state), 0) # size=(inp_size)


class Hopfield(nn.Module):
    def __init__(self, X):
        super().__init__()
        if X.dim() == 1:
            X = X.view(-1, 1)
        self.X = X  # (inp_size, num_patterns)
        self.N = torch.tensor(X.shape[1])  # number of patterns
        self.M = torch.max(X)

    def energy_fn(self, state):
        lse = -torch.logsumexp(torch.matmul(self.X.T, state), 0) # float
        quad = torch.dot(state, state) # float
        divers = B * torch.log(self.N) + self.M**2 * 0.5 # float

        E = lse + quad + divers # float
        return E

    def forward(self, state):
        new_state = self.X @ torch.softmax(B * torch.matmul(self.X.T, state), 0) # (inp_size)
        return new_state
import torch
import torch.nn as nn
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from image_processing import tensor_to_image, process_image
from torchvision import datasets, transforms

inp_size = 64*64

N = torch.tensor(1)
X = torch.randn(inp_size, N)
B = 1

M = torch.max(X).item()
state = torch.randn(inp_size)

lse = -torch.logsumexp(torch.matmul(X.T, state), 0) # return a float
state_mul = torch.dot(state, state)*0.5 # return a float
divers = B * torch.log(N) + M**2 * 0.5
E = lse + state_mul + divers
new_state = X @ torch.softmax(torch.matmul(X.T, state), 0) # size=(inp_size)


class Hopfield(nn.Module):
    def __init__(self, X):
        self.X = X # (inp_size, num_patterns)
        self.N = torch.tensor(X.shape[1]) # number of patterns
        self.M = torch.max(X)

    def energy_fn(self, state):
        lse = -torch.logsumexp(torch.matmul(self.X.T, state), 0) # float
        quad = torch.dot(state, state) # float
        divers = torch.log(self.N) + self.M**2 * 0.5 # float

        E = lse + quad + divers # float
        return E

    def forward(self, state):
        new_state = self.X @ torch.softmax(torch.matmul(self.X.T, state), 0) # (inp_size)
        return new_state


test_image = process_image('test1.png')
# print(test_image.shape)
# print(test_image)
hopfield = Hopfield(test_image)
state = torch.randn(test_image.shape)
# tensor_to_image(state)

noisy_pattern = test_image + 0.2 * torch.randn_like(test_image)
# tensor_to_image(test_image)
# tensor_to_image(noisy_pattern)

state = hopfield.forward(state)
state = hopfield.forward(state)
state = hopfield.forward(state)
state = hopfield.forward(state)
state = hopfield.forward(state)

plt.imshow(state, cmap='gray')
plt.show()
# tensor_to_image(state)

data = datasets.MNIST('../data', train=True, download=True)
imgs = torch.stack([transforms.functional.pil_to_tensor(data[i][0]).flatten().float() for i in range(300)], 1)
# print(imgs.shape)
# print(imgs)
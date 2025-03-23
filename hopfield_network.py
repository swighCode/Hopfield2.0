import torch
import torch.nn as nn
import numpy as np
import sklearn as sk
import os
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
        if X.dim() == 1:
            X = X.view(-1, 1)
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


test_image = process_image('testface1.jpg')
test_image_2 = process_image('testface2.jpg')
test_image_3 = process_image('testface3.jpg')
'''
hopfield = Hopfield(torch.stack((test_image, test_image_2)).T)
state = torch.randn(test_image_3.shape)
# tensor_to_image(state)

noisy_pattern = test_image_3 + 0.5 * torch.randn_like(test_image_3)
censored_face = process_image('censoredface.jpg')
tensor_to_image(test_image_3)
tensor_to_image(censored_face)

state = hopfield.forward(censored_face)
tensor_to_image(state)
'''
#Pseudo code for hopfield using dataset:
query_images = [test_image_3] #Query images (face has to be in network, but not identical image)
#query_images = [qimg_1, qimg_2, qimg_3, qimg_4, qimg_5]
query_imgs_in_network = [test_image] #The corresponding faces we want to retrieve (fill manually?)
#query_imgs_in_network = [qiin_1, qiin_2, qiin_3, qiin_4, qiin_5]
loaded_images = torch.load("faces_dataset.pt") #Array of image tensors that we later stack.
amount_epochs = 10
epoch = 0
failed_epochs = []
for query_image in query_images:
    while epoch < amount_epochs:
        print("Epoch {}/{}".format(epoch+1, amount_epochs))
        curr_batch = torch.stack(query_imgs_in_network)
        for i in range(int(len(loaded_images)*epoch/amount_epochs), int(len(loaded_images)*(epoch+1)/amount_epochs)):
            curr_batch = torch.stack((curr_batch, loaded_images[i]))
        hopfield_network = Hopfield(curr_batch.T)
        result = hopfield_network(query_image)
        tensor_to_image(query_image)
        tensor_to_image(result)
        if not np.isclose(result, curr_batch[query_image.index()]):
            print("Failed to retrieve at epoch {}/{}".format(epoch+1, amount_epochs))
            failed_epochs.append(epoch)
            break
        else:
            print("Successfully retrieved at epoch {}/{}".format(epoch+1, amount_epochs))
            epoch = epoch + 1

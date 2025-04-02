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
B = .0001

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



def main():
    # Load and process training images with labels
    training_images_folder = 'dataset'
    processed_images = []

    for root, _, files in os.walk(training_images_folder):
        label = os.path.basename(root)
        for file in files:
            split = file.split('_')
            if file.endswith('.jpg') and split[1].startswith('2') and "30" in split[2]:
                image_path = os.path.join(root, file)
                image = process_image(image_path)
                processed_images.append((image, split[0]))

    # Save and load with labels
    torch.save(processed_images, 'faces_dataset.pt')
    loaded_list = torch.load("faces_dataset.pt", weights_only=True)
    stored_images = [item[0] for item in loaded_list]
    stored_labels = [item[1] for item in loaded_list]
    stored_patterns = torch.stack(stored_images).T

    # Initialize Hopfield network
    hopnet = Hopfield(stored_patterns)
    print(f"Network initialized with {stored_patterns.shape[1]} patterns")

    # Load query images with labels
    query_data = []
    test_faces_folder = 'dataset'
    for root, _, files in os.walk(test_faces_folder):
        label = os.path.basename(root)
        for file in files:
            split = file.split('_')
            if file.endswith('.jpg') and split[1].startswith('22') and "00" in split[2]:
                print(file)
                image_path = os.path.join(root, file)
                image = process_image(image_path)
                query_data.append((image, split[0], file))

    # Set display flag to true for visualization
    display = True

    num_success = 0
    for query_img, query_label, query_name in query_data:
        print(f"\nTesting retrieval for: {query_name} (True class: {query_label})")
        
        # Add noise to the query pattern
        noisy_query = query_img + 2 * torch.randn_like(query_img)
        
        # Display original and noisy
        if display:
            tensor_to_image(query_img)
            tensor_to_image(noisy_query)
        
        # Retrieve pattern through network dynamics
        state = noisy_query.clone()
        prev_state = torch.zeros_like(state)
        
        # Iterate until convergence or max iterations
        for i in range(100):
            prev_state.copy_(state)
            state = hopnet.forward(state)
            # tensor_to_image(state)
            if torch.allclose(state, prev_state, atol=1e-4):
                break
        
        # Show the network's actual output
        if display:
            tensor_to_image(state)
        
        # Find closest stored pattern
        similarity = torch.matmul(stored_patterns.T, state)
        retrieved_idx = torch.argmax(similarity)
        retrieved_label = stored_labels[retrieved_idx]
        
        # Check if retrieved label matches query's true label
        if retrieved_label == query_label:
            num_success += 1
            print(f"Retrieved class: {retrieved_label} ✔️")
        else:
            print(f"Retrieved class: {retrieved_label} ❌")

    print(f"\nRetrieval accuracy: {num_success/len(query_data):.2%} {len(query_data)} images tested")

if __name__ == "__main__":
    main()
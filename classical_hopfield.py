import torch
import torch.nn as nn

class ClassicalHopfield(nn.Module):
    def __init__(self, X_patterns):
        # X_patterns: Tensor of shape (num_neurons, num_patterns)
        # Each column is a bipolar (+1/-1) pattern.
        super().__init__()

        if X_patterns.dim() == 1:
            # If a single 1D pattern is given, reshape to (num_neurons, 1)
            X_patterns = X_patterns.view(-1, 1)
        
        # Assert input tensor properties
        assert X_patterns.dim() == 2, "Input X_patterns must be a 2D tensor."
        assert torch.all((X_patterns == 1) | (X_patterns == -1)), \
               "Input patterns must be bipolar (+1/-1)."

        self.num_neurons = X_patterns.size(0)
        self.num_patterns = X_patterns.size(1)

        # Compute the weight matrix using Hebbian rule: W = X @ X.T
        # Normalization can be by num_neurons, num_patterns, or none.
        # Here, we demonstrate normalization by num_patterns as one common choice.
        # For the user's original normalization by num_neurons:
        # W = torch.mm(X_patterns, X_patterns.t()) / self.num_neurons
        if self.num_patterns > 0:
             W = torch.mm(X_patterns, X_patterns.t()) / self.num_patterns
        else: # Avoid division by zero if no patterns are provided
             W = torch.zeros((self.num_neurons, self.num_neurons), 
                             device=X_patterns.device, dtype=X_patterns.dtype)

        # Zero the diagonal elements: W_ii = 0
        # This prevents self-connections and is crucial for stability.
        W.fill_diagonal_(0)
        
        # Register W as a buffer (part of state, not a trainable parameter)
        self.register_buffer("W", W)

    def forward(self, state, num_iterations=1):
        # state: The initial state vector (num_neurons), bipolar (+1/-1).
        # num_iterations: Number of full asynchronous update passes.
        
        assert state.dim() == 1, "Initial state must be a 1D tensor."
        assert torch.all((state == 1) | (state == -1)), \
               "Initial state must be bipolar (+1/-1)."
        assert state.size(0) == self.num_neurons, \
               "Initial state dimension must match num_neurons."

        new_state = state.clone()
        for _ in range(num_iterations):
            # Asynchronously update each neuron in random order for one full pass
            indices = torch.randperm(self.num_neurons, device=state.device)
            for i in indices:
                # Calculate activation: h_i = sum_j(W_ij * s_j)
                # self.W[i] is the i-th row of W.
                activation = torch.dot(self.W[i], new_state)
                # Apply signum activation function (threshold = 0)
                new_state[i] = 1.0 if activation >= 0 else -1.0
        return new_state

    def energy(self, state):
        # Calculates the energy of a given state for this Hopfield network.
        # E = -0.5 * state.T @ W @ state
        # Note: some definitions omit the 0.5, or use -state @ W @ state.
        # The 0.5 factor is common in physics-based formulations.
        # For W_ii = 0, E = - sum_{i<j} W_ij s_i s_j
        
        assert state.dim() == 1, "State must be a 1D tensor."
        assert torch.all((state == 1) | (state == -1)), \
               "State must be bipolar (+1/-1)."
        assert state.size(0) == self.num_neurons, \
               "State dimension must match num_neurons."
        
        if self.W is None or self.W.numel() == 0:
            # This case should ideally not happen if patterns were provided during init.
            # If num_patterns was 0, W is all zeros.
            return torch.tensor(0.0, device=state.device, dtype=state.dtype)
        
        # For E = -0.5 * s^T W s, if s is 1D:
        # term1 = W @ s (matrix-vector product)
        term1 = torch.mv(self.W, state) 
        # E = -0.5 * s. (W @ s) (dot product)
        E = -0.5 * torch.dot(state, term1)
        return E
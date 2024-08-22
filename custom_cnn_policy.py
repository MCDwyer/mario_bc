import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom CNN Feature Extractor
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, trial, features_dim=512):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Sample hyperparameters
        self.n_layers = trial.suggest_int('n_layers', 2, 5)
        self.n_filters = trial.suggest_categorical('n_filters', [16, 32, 64, 128])
        self.activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'selu'])
        self.use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

        layers = []
        in_channels = n_input_channels
        current_dim = observation_space.shape[1]  # Assuming square inputs, otherwise use shape[1] and shape[2]

        for i in range(self.n_layers):
            # Adjust kernel size to ensure it fits within the current dimensions
            max_kernel_size = min(5, current_dim)
            self.kernel_size = trial.suggest_int('kernel_size', 2, max_kernel_size)
            self.stride = trial.suggest_int('stride', 1, 2)
            self.padding = trial.suggest_int('padding', 0, 2)


            out_channels = self.n_filters * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # Use a dictionary to map the activation function string to the actual function
            activation_function = {
                'relu': nn.ReLU(),
                'leaky_relu': nn.LeakyReLU(),
                'elu': nn.ELU(),
                'selu': nn.SELU()
            }[self.activation]
            
            layers.append(activation_function)
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

            # Calculate the output size after this layer
            output_size = (28 - (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
            output_size = (output_size - 2) // 2 + 1  # After MaxPool
            if output_size <= 0:
                raise ValueError(f"Output size too small after layer {i+1}. Consider adjusting kernel size, stride, or padding.")

        self.cnn = nn.Sequential(*layers)

        # Compute the size of the output of the CNN to feed into the fully connected layers
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.shape[1] * cnn_output.shape[2] * cnn_output.shape[3]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

    

def train_cnn(expert_observations, expert_actions, num_epochs, batch_size, model, loss_fn, optimiser):

    for epoch in range(num_epochs):
        # Shuffle data at the start of each epoch
        permutation = np.random.permutation(len(expert_observations))
        expert_observations = expert_observations[permutation]
        expert_actions = expert_actions[permutation]
        
        for i in range(0, len(expert_observations), batch_size):
            batch_obs = expert_observations[i:i + batch_size]
            batch_actions = expert_actions[i:i + batch_size]

            # Forward pass: compute predicted actions by passing observations to the policy
            logits, _, _ = model.policy(batch_obs)

            # Calculate loss
            loss = loss_fn(logits.float(), batch_actions.float())

            # Backward pass: compute gradient of the loss with respect to model parameters
            optimiser.zero_grad()
            loss.backward(retain_graph=True)

            # Update model parameters
            optimiser.step()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

    return model
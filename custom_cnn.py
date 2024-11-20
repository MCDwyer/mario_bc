# import torch as th
# import torch.nn as nn
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3 import PPO
# from stable_baselines3.common.policies import ActorCriticCnnPolicy
# import gymnasium as gym
# from gym import spaces
# # import gym

# class CustomCNN(BaseFeaturesExtractor):
#     """
#     Custom CNN feature extractor for the policy network.
#     """
#     def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
#         # Call the BaseFeaturesExtractor constructor
#         super(CustomCNN, self).__init__(observation_space, features_dim)
        
#         # Define the CNN architecture
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten()
#         )
        
#         # Compute the size of the flattened features after the CNN layers
#         with th.no_grad():
#             # Dummy forward pass to get the output size of the CNN
#             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
#         # Final fully connected layer to reduce to the desired feature size (features_dim)
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         # Pass the observations through the CNN and then the linear layer
#         return self.linear(self.cnn(observations))

# class CustomCnnPolicy(ActorCriticCnnPolicy):
#     def __init__(self, *args, **kwargs):
#         # Use the custom CNN as the feature extractor
#         super(CustomCnnPolicy, self).__init__(
#             *args, 
#             **kwargs,
#             features_extractor_class=CustomCNN,
#             features_extractor_kwargs=dict(features_dim=512)  # Pass feature dimension
#         )


import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

DIMENSIONS = 512
class CustomNatureCNN(nn.Module):
    def __init__(self, input_channels=4, output_dim=512):
        super(CustomNatureCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the output shape after convolutions for the fully connected layer
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 84, 84)
            conv_out = self._forward_cnn(sample_input)
            self.fc_input_dim = conv_out.view(-1).shape[0]
        
        # Fully connected layer for feature extraction
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def _forward_cnn(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_cnn(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = F.relu(self.fc(x))
        return x

class CustomMLPExtractor(nn.Module):
    def __init__(self, input_dim=512):
        super(CustomMLPExtractor, self).__init__()
        
        # MLP layers for policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, DIMENSIONS),
            nn.ReLU()
        )
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, DIMENSIONS),
            nn.ReLU()
        )

    def forward(self, x):
        policy_features = self.policy_net(x)
        value_features = self.value_net(x)
        return policy_features, value_features

class CustomCnnPolicy(nn.Module):
    def __init__(self, env):
        super(CustomCnnPolicy, self).__init__()

        # Check the observation space
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            # Typically for image observations, the shape will be (Height, Width, Channels)
            input_channels = obs_shape[2]
        else:
            # If it's a flat observation space, the whole shape can be used
            input_channels = obs_shape[0]  # for non-image environments, like CartPole

        # Check the action space
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_space = env.action_space.n  # Number of discrete actions
        else:
            action_space = env.action_space.shape[0]  # for continuous actions (like in MuJoCo)

        input_dim = 512
        # Feature extractor
        self.features_extractor = CustomNatureCNN(input_channels, output_dim=input_dim)
        
        # MLP layers for policy and value
        self.mlp_extractor = CustomMLPExtractor(input_dim=input_dim)
        
        # Action and value heads
        self.action_net = nn.Linear(input_dim, action_space)
        self.value_net = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Extract features from CNN
        features = self.features_extractor(x)
        
        # Process features with MLP for policy and value
        policy_features, value_features = self.mlp_extractor(features)
        
        # Final policy and value outputs
        action_logits = self.action_net(policy_features)
        value = self.value_net(value_features)
        
        return action_logits, value

    def predict(self, x):
        action_logits, value = self.forward(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, value

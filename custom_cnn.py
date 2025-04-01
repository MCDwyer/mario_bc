import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.dqn.policies import CnnPolicy

from gymnasium import spaces
import numpy as np
import copy

class CustomDQNPolicy(CnnPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self, rl_agent):
        rl_policy = rl_agent.policy
        
        super().__init__(
            rl_policy.observation_space,
            rl_policy.action_space,
            rl_agent.lr_schedule,
            rl_policy.net_arch,
            rl_policy.activation_fn,
            rl_policy.features_extractor_class,
            rl_policy.features_extractor_kwargs,
            rl_policy.normalize_images,
            rl_policy.optimizer_class,
            rl_policy.optimizer_kwargs,
            )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def bc_actions(self, obs):

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            # latent_vf = self.mlp_extractor.forward_critic(vf_features)

        mean_actions = self.action_net(latent_pi)

        return mean_actions

    def bc_training(self, observations, actions, num_epochs, batch_size, device="cpu") -> None:

        # Expert data should not require gradients
        expert_actions = torch.tensor(actions, dtype=torch.long).to(device)
        expert_observations = torch.tensor(observations, dtype=torch.float32).to(device)

        dataset_size = len(observations)

        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)
        
        for epoch in range(num_epochs):
            # Shuffle data at the start of each epoch
            permutation = np.random.permutation(len(expert_observations))
            expert_obs = copy.deepcopy(expert_observations[permutation])
            expert_acts = copy.deepcopy(expert_actions[permutation])

            epoch_loss = 0.0
            num_batches = int(np.ceil(dataset_size / batch_size))

            # Mini-batch training
            for i in range(num_batches):
                # batch_indices = np.random.choice(dataset_size, batch_size)
                obs_batch = expert_obs[:batch_size]
                action_batch = expert_acts[:batch_size]

                # Forward pass
                pred_actions = self.bc_actions(obs_batch)

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    action_batch = action_batch.long().flatten()
                    # pred_actions = pred_actions.long().flatten()

                loss = self.criterion(pred_actions, action_batch)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                expert_obs = expert_obs[batch_size:]
                expert_acts = expert_acts[batch_size:]

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")

        


class CustomActorCriticCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, rl_agent):
        
        # Policy stuff:

        # observation_space: spaces.Space,
        # action_space: spaces.Space,
        # lr_schedule: Schedule,
        # net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        # activation_fn: type[nn.Module] = nn.Tanh,
        # ortho_init: bool = True,
        # use_sde: bool = False,
        # log_std_init: float = 0.0,
        # full_std: bool = True,
        # use_expln: bool = False,
        # squash_output: bool = False,
        # features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        # features_extractor_kwargs: Optional[dict[str, Any]] = None,
        # share_features_extractor: bool = True,
        # normalize_images: bool = True,
        # optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        # optimizer_kwargs: Optional[dict[str, Any]] = None,
        
        # PPO stuff:
        # policy: Union[str, type[ActorCriticPolicy]],
        # env: Union[GymEnv, str],
        # learning_rate: Union[float, Schedule] = 3e-4,
        # n_steps: int = 2048,
        # batch_size: int = 64,
        # n_epochs: int = 10,
        # gamma: float = 0.99,
        # gae_lambda: float = 0.95,
        # clip_range: Union[float, Schedule] = 0.2,
        # clip_range_vf: Union[None, float, Schedule] = None,
        # normalize_advantage: bool = True,
        # ent_coef: float = 0.0,
        # vf_coef: float = 0.5,
        # max_grad_norm: float = 0.5,
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        # rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        # rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        # target_kl: Optional[float] = None,
        # stats_window_size: int = 100,
        # tensorboard_log: Optional[str] = None,
        # policy_kwargs: Optional[dict[str, Any]] = None,
        # verbose: int = 0,
        # seed: Optional[int] = None,
        # device: Union[th.device, str] = "auto",
        # _init_setup_model: bool = True,

        rl_policy = rl_agent.policy

        super().__init__(
            rl_policy.observation_space,
            rl_policy.action_space,
            rl_agent.lr_schedule,
            rl_policy.net_arch,
            rl_policy.activation_fn,
            rl_policy.ortho_init,
            rl_policy.use_sde,
            rl_policy.log_std_init,
            True,
            False,
            rl_policy.squash_output,
            rl_policy.features_extractor_class,
            rl_policy.features_extractor_kwargs,
            rl_policy.share_features_extractor,
            rl_policy.normalize_images,
            rl_policy.optimizer_class,
            rl_policy.optimizer_kwargs,
        )

        self.criterion = nn.CrossEntropyLoss()
    
    def bc_actions(self, obs):

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            # latent_vf = self.mlp_extractor.forward_critic(vf_features)

        mean_actions = self.action_net(latent_pi)

        return mean_actions

    def bc_training(self, observations, actions, num_epochs, batch_size, device="cpu") -> None:

        # Handles lists, NumPy arrays, or tensors
        first_action = expert_actions[0]

        if isinstance(first_action, (list, np.ndarray, torch.Tensor)) and len(first_action) > 1:
            use_probs = True
            print("Using probabilities over actions for training.")

        else:
            use_probs = False
            print("Using discrete actions for training.")

        if use_probs:
            expert_actions = torch.tensor(actions, dtype=torch.float32).to(device)  # shape: [N, 13]
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        else:
            expert_actions = torch.tensor(actions, dtype=torch.long).to(device)  # shape: [N]
            self.criterion = nn.CrossEntropyLoss()

        # Expert data should not require gradients
        # expert_actions = torch.tensor(actions, dtype=torch.long).to(device)
        expert_observations = torch.tensor(observations, dtype=torch.float32).to(device)

        dataset_size = len(observations)

        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)
        
        for epoch in range(num_epochs):
            # Shuffle data at the start of each epoch
            permutation = np.random.permutation(len(expert_observations))
            expert_obs = copy.deepcopy(expert_observations[permutation])
            expert_acts = copy.deepcopy(expert_actions[permutation])

            epoch_loss = 0.0
            num_batches = int(np.ceil(dataset_size / batch_size))

            # Mini-batch training
            for i in range(num_batches):
                # batch_indices = np.random.choice(dataset_size, batch_size)
                obs_batch = expert_obs[:batch_size]
                action_batch = expert_acts[:batch_size]

                if use_probs:
                    dist = self.get_distribution(obs_batch)
                    pred_probs = dist.distribution.probs
                    pred_actions = torch.log(pred_probs + 1e-8)
                else:
                    # Forward pass
                    pred_actions = self.bc_actions(obs_batch)

                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        action_batch = action_batch.long().flatten()

                loss = self.criterion(pred_actions, action_batch)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                expert_obs = expert_obs[batch_size:]
                expert_acts = expert_acts[batch_size:]

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")



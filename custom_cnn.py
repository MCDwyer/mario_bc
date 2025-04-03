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

        
class OfflineReplayBuffer:
    def __init__(self, observations, actions, rewards, next_observations, dones):
        self.obs = torch.tensor(observations, dtype=torch.float32)
        self.acts = torch.tensor(actions, dtype=torch.long)
        self.rews = torch.tensor(rewards, dtype=torch.float32)
        self.next_obs = torch.tensor(next_observations, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.float32)

    def sample(self, batch_size):
        idxs = torch.randint(0, len(self.obs), (batch_size,))
        return (
            self.obs[idxs],
            self.acts[idxs],
            self.rews[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
        )

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

        self.bc_criterion = nn.CrossEntropyLoss()
        self.bc_probs_criterion = nn.KLDivLoss(reduction="batchmean")

        self.v_criterion = nn.MSELoss()

    
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
        first_action = actions[0]

        if isinstance(first_action, (list, np.ndarray, torch.Tensor)) and len(first_action) > 1:
            use_probs = True
            print("Using probabilities over actions for training.")

        else:
            use_probs = False
            print("Using discrete actions for training.")

        if use_probs:
            expert_actions = torch.tensor(actions, dtype=torch.float32).to(device)  # shape: [N, 13]
            criterion = self.bc_probs_criterion
        else:
            expert_actions = torch.tensor(actions, dtype=torch.long).to(device)  # shape: [N]
            criterion = self.bc_criterion

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

                loss = criterion(pred_actions, action_batch)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                expert_obs = expert_obs[batch_size:]
                expert_acts = expert_acts[batch_size:]

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")


    def offline_actor_critic_training(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        lr=1e-5,
        batch_size=64,
        num_epochs=20,
        device="cpu",
        gamma=0.99,
        reward_scale=1.0,
        entropy_coef=0.01,
        target_update_freq=100
    ):
        # --- Preprocess ---
        # Convert to float tensors, move to device
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device) / reward_scale
        dones = torch.tensor(dones, dtype=torch.float32).to(device)  # no gradients needed
    
        # Setup replay buffer
        replay_buffer = OfflineReplayBuffer(observations, actions, rewards, next_observations, dones)

        self.to(device)
        self.set_training_mode(True)

        # Copy target value net
        target_value_net = copy.deepcopy(self.value_net)
        target_value_net.eval().to(device)

        optimizer_pi = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer_v = torch.optim.Adam(self.parameters(), lr=lr)

        step_count = 0

        for epoch in range(num_epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            num_batches = len(replay_buffer.obs) // batch_size

            for _ in range(num_batches):
                step_count += 1

                obs, act, rew, next_obs, done = replay_buffer.sample(batch_size)
                obs = obs.to(device)
                act = act.to(device)
                rew = rew.to(device)
                next_obs = next_obs.to(device)
                done = done.to(device)

                # --- Critic update ---
                with torch.no_grad():
                    next_values = target_value_net(self.extract_features(next_obs)).squeeze(1)
                    target_values = rew + gamma * (1.0 - done) * next_values

                predicted_values = self.value_net(self.extract_features(obs)).squeeze(1)
                critic_loss = self.v_criterion(predicted_values, target_values)

                optimizer_v.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer_v.step()

                # --- Advantage estimation ---
                with torch.no_grad():
                    advantages = target_values - predicted_values.detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    advantages = torch.clamp(advantages, -10, 10)  # optional clipping

                # --- Actor update ---
                dist = self.get_distribution(obs)
                log_probs = dist.log_prob(act)
                entropy = dist.entropy()

                actor_loss = -(log_probs * advantages).mean() - entropy_coef * entropy.mean()

                optimizer_pi.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer_pi.step()

                # --- Update target net periodically ---
                if step_count % target_update_freq == 0:
                    target_value_net.load_state_dict(self.value_net.state_dict())

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()

            print(f"Epoch {epoch+1} | Actor Loss: {epoch_actor_loss / num_batches:.4f} | Critic Loss: {epoch_critic_loss / num_batches:.4f}")

    def bc_with_value_training(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        num_epochs=10,
        batch_size=64,
        device="cpu",
        gamma=0.99,
        lr=1e-4
    ):
        # Prepare tensors
        obs = torch.tensor(observations, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_obs = torch.tensor(next_observations, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        self.to(device)
        self.set_training_mode(True)

        optimizer_pi = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer_v = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        dataset_size = len(obs)

        for epoch in range(num_epochs):
            permutation = torch.randperm(dataset_size)
            epoch_pi_loss = 0.0
            epoch_v_loss = 0.0

            for i in range(0, dataset_size, batch_size):
                idxs = permutation[i:i+batch_size]
                obs_batch = obs[idxs]
                act_batch = actions[idxs]
                rew_batch = rewards[idxs]
                next_obs_batch = next_obs[idxs]
                done_batch = dones[idxs]

                # --- Policy (BC) loss ---
                logits = self.bc_actions(obs_batch)  # raw logits
                pi_loss = self.bc_criterion(logits, act_batch)

                optimizer_pi.zero_grad()
                pi_loss.backward()
                optimizer_pi.step()

                # --- Value function loss ---
                with torch.no_grad():
                    next_values = self.value_net(self.extract_features(next_obs_batch)).squeeze(1)
                    targets = rew_batch + gamma * (1.0 - done_batch) * next_values

                values = self.value_net(self.extract_features(obs_batch)).squeeze(1)
                v_loss = self.v_criterion(values, targets)

                optimizer_v.zero_grad()
                v_loss.backward()
                optimizer_v.step()

                epoch_pi_loss += pi_loss.item()
                epoch_v_loss += v_loss.item()

            print(f"Epoch {epoch+1} | BC Loss: {epoch_pi_loss:.4f} | Value Loss: {epoch_v_loss:.4f}")

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

    # def offline_rl_training(self, observations, actions, rewards, num_epochs, batch_size, device="cpu"):

    #     print("temp")
    #     return
    
    # def a2c_offline_rl_training(self, model, observations, actions, rewards, num_epochs, batch_size, device="cpu"):
    #     """
    #     Update policy using the currently gathered
    #     rollout buffer (one gradient step over whole data).
    #     """
    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.set_training_mode(True)

    #     # Update optimizer learning rate
    #     model._update_learning_rate(self.optimizer)

    #    # Expert data should not require gradients
    #     expert_actions = torch.tensor(actions, dtype=torch.long).to(device)
    #     expert_observations = torch.tensor(observations, dtype=torch.float32).to(device)
    #     expert_rewards = torch.tensor(rewards).to(device)

    #     dataset_size = len(observations)

    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.set_training_mode(True)
        
    #     for epoch in range(num_epochs):
    #         # Shuffle data at the start of each epoch
    #         permutation = np.random.permutation(len(expert_observations))
    #         expert_obs = copy.deepcopy(expert_observations[permutation])
    #         expert_acts = copy.deepcopy(expert_actions[permutation])

    #         epoch_loss = 0.0
    #         num_batches = int(np.ceil(dataset_size / batch_size))

    #         # Mini-batch training
    #         for i in range(num_batches):
    #             # batch_indices = np.random.choice(dataset_size, batch_size)
    #             obs_batch = expert_obs[:batch_size]
    #             action_batch = expert_acts[:batch_size]

    #             # Forward pass
    #             pred_actions = self.bc_actions(obs_batch)

    #             if isinstance(self.action_space, spaces.Discrete):
    #                 # Convert discrete action from float to long
    #                 action_batch = action_batch.long().flatten()
    #                 # pred_actions = pred_actions.long().flatten()

    #             loss = self.criterion(pred_actions, action_batch)

    #             # Optimization step
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             epoch_loss += loss.item()

    #             expert_obs = expert_obs[batch_size:]
    #             expert_acts = expert_acts[batch_size:]

    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")


    #     # This will only loop once (get all data in one go)
    #     for rollout_data in self.rollout_buffer.get(batch_size=None):
    #         actions = rollout_data.actions
    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Convert discrete action from float to long
    #             actions = actions.long().flatten()

    #         values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)
    #         values = values.flatten()

    #         # Normalize advantage (not present in the original implementation)
    #         advantages = rollout_data.advantages
    #         if self.normalize_advantage:
    #             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #         # Policy gradient loss
    #         policy_loss = -(advantages * log_prob).mean()

    #         # Value loss using the TD(gae_lambda) target
    #         value_loss = F.mse_loss(rollout_data.returns, values)

    #         # Entropy loss favor exploration
    #         if entropy is None:
    #             # Approximate entropy when no analytical form
    #             entropy_loss = -th.mean(-log_prob)
    #         else:
    #             entropy_loss = -th.mean(entropy)

    #         loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    #         # Optimization step
    #         self.optimizer.zero_grad()
    #         loss.backward()

    #         # Clip grad norm
    #         th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    #         self.optimizer.step()
    
        
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     n_steps = 0
    #     rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.policy.reset_noise(env.num_envs)

    #     callback.on_rollout_start()

    #     while n_steps < n_rollout_steps:
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.policy.reset_noise(env.num_envs)

    #         with th.no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)
    #             actions, values, log_probs = self.policy(obs_tensor)
    #         actions = actions.cpu().numpy()

    #         # Rescale and perform action
    #         clipped_actions = actions

    #         if isinstance(self.action_space, spaces.Box):
    #             if self.policy.squash_output:
    #                 # Unscale the actions to match env bounds
    #                 # if they were previously squashed (scaled in [-1, 1])
    #                 clipped_actions = self.policy.unscale_action(clipped_actions)
    #             else:
    #                 # Otherwise, clip the actions to avoid out of bound error
    #                 # as we are sampling from an unbounded Gaussian distribution
    #                 clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         new_obs, rewards, dones, infos = env.step(clipped_actions)

    #         self.num_timesteps += env.num_envs

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         if not callback.on_step():
    #             return False

    #         self._update_info_buffer(infos, dones)
    #         n_steps += 1

    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)

    #         # Handle timeout by bootstrapping with value function
    #         # see GitHub issue #633
    #         for idx, done in enumerate(dones):
    #             if (
    #                 done
    #                 and infos[idx].get("terminal_observation") is not None
    #                 and infos[idx].get("TimeLimit.truncated", False)
    #             ):
    #                 terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #                 with th.no_grad():
    #                     terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
    #                 rewards[idx] += self.gamma * terminal_value

    #         rollout_buffer.add(
    #             self._last_obs,  # type: ignore[arg-type]
    #             actions,
    #             rewards,
    #             self._last_episode_starts,  # type: ignore[arg-type]
    #             values,
    #             log_probs,
    #         )
    #         self._last_obs = new_obs  # type: ignore[assignment]
    #         self._last_episode_starts = dones

    #     with th.no_grad():
    #         # Compute value for the last timestep
    #         values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

    #     rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    #     callback.update_locals(locals())

    #     callback.on_rollout_end()

    
    # def ppo_offline_rl_training(self, model, observations, actions, rewards, num_epochs, batch_size, device="cpu"):
    #     """
    #     Update policy using the currently gathered rollout buffer.
    #     """

    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.set_training_mode(True)
        
    #     # Update optimizer learning rate
    #     model._update_learning_rate(self.optimizer)
    #     # Compute current clip range
    #     clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    #     # Optional: clip range for the value function
    #     if self.clip_range_vf is not None:
    #         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    #     entropy_losses = []
    #     pg_losses, value_losses = [], []
    #     clip_fractions = []

    #     continue_training = True
    #     # train for n_epochs epochs
    #     for epoch in range(self.n_epochs):
    #         approx_kl_divs = []
    #         # Do a complete pass on the rollout buffer
    #         for rollout_data in self.rollout_buffer.get(self.batch_size):
    #             actions = rollout_data.actions
    #             if isinstance(self.action_space, spaces.Discrete):
    #                 # Convert discrete action from float to long
    #                 actions = rollout_data.actions.long().flatten()

    #             values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
    #             values = values.flatten()
    #             # Normalize advantage
    #             advantages = rollout_data.advantages
    #             # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    #             if self.normalize_advantage and len(advantages) > 1:
    #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #             # ratio between old and new policy, should be one at the first iteration
    #             ratio = th.exp(log_prob - rollout_data.old_log_prob)

    #             # clipped surrogate loss
    #             policy_loss_1 = advantages * ratio
    #             policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    #             policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    #             # Logging
    #             pg_losses.append(policy_loss.item())
    #             clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
    #             clip_fractions.append(clip_fraction)

    #             if self.clip_range_vf is None:
    #                 # No clipping
    #                 values_pred = values
    #             else:
    #                 # Clip the difference between old and new value
    #                 # NOTE: this depends on the reward scaling
    #                 values_pred = rollout_data.old_values + th.clamp(
    #                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf
    #                 )
    #             # Value loss using the TD(gae_lambda) target
    #             value_loss = F.mse_loss(rollout_data.returns, values_pred)
    #             value_losses.append(value_loss.item())

    #             # Entropy loss favor exploration
    #             if entropy is None:
    #                 # Approximate entropy when no analytical form
    #                 entropy_loss = -th.mean(-log_prob)
    #             else:
    #                 entropy_loss = -th.mean(entropy)

    #             entropy_losses.append(entropy_loss.item())

    #             loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    #             # Calculate approximate form of reverse KL Divergence for early stopping
    #             # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
    #             # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
    #             # and Schulman blog: http://joschu.net/blog/kl-approx.html
    #             with th.no_grad():
    #                 log_ratio = log_prob - rollout_data.old_log_prob
    #                 approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
    #                 approx_kl_divs.append(approx_kl_div)

    #             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
    #                 continue_training = False
    #                 if self.verbose >= 1:
    #                     print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
    #                 break

    #             # Optimization step
    #             self.policy.optimizer.zero_grad()
    #             loss.backward()
    #             # Clip grad norm
    #             th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #             self.policy.optimizer.step()

    #         self._n_updates += 1
    #         if not continue_training:
    #             break

    #     explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

import torch
import torch.nn as nn
import numpy as np
import pickle

# from imitation.data.types import Trajectory, Transitions
# from imitation.data import rollout
# from imitation.algorithms import bc

from custom_cnn_policy import train_cnn


NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# def behavioural_cloning_with_imitation(env, model_path, training_filepath):

#     # load in data
#     with open(training_filepath, 'rb') as file:
#         trajectories = pickle.load(file)

#     trajectories = np.array(trajectories)

#     infos = [{} for _ in range(len(trajectories))]  # Empty dicts, assuming no additional info is available

#     observations = trajectories[:, 0]
#     actions = trajectories[:, 1]
#     done = np.array(trajectories[:, 2], dtype=bool)
#     next_observations = trajectories[:, 3]

#     # Convert expert data into Trajectories or Transitions
#     transitions = Transitions(obs=observations, acts=actions, infos=infos, next_obs=next_observations, dones=done)

#     # Initialize the BC algorithm
#     bc_trainer = bc.BC(
#         expert_data=transitions,
#         # policy_class=MODEL_CLASS,  # The policy class to be used
#         # policy_kwargs=dict(policy=POLICY),  # Optional: modify network architecture
#         # env=env,
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#     )

#     # Train the model using behavior cloning
#     bc_trainer.train(n_epochs=NUM_EPOCHS)

#     bc_model_path = f"{model_path}_bc"

#     # Save the trained model
#     bc_trainer.policy.save(bc_model_path)

#     return bc_trainer, bc_model_path

# def preprocess_observations(observations):
#     # Normalize observations (e.g., images with values between 0 and 255)
#     observations = observations / 255.0

#     # Ensure the shape is in the format (batch_size, channels, height, width)
#     if len(observations.shape) == 3:
#         observations = np.expand_dims(observations, axis=0)  # Add batch dimension
#     if observations.shape[1] != 3:  # Assuming color images
#         observations = np.transpose(observations, (0, 3, 1, 2))  # Convert from HWC to CHW

#     return observations


def load_data(filepath):
    # load in data
    with open(filepath, 'rb') as file:
        trajectories = pickle.load(file)

    trajectories = np.array(trajectories, dtype=object)

    # for some reason have to do it this way, otherwise it gets upset
    observations = [np.array(obs, dtype=np.float32) for obs in trajectories[:, 0]]
    observations = np.array(observations)
    print(observations.shape)
    # observations = preprocess_observations(observations)

    actions = np.array(trajectories[:, 1], dtype=np.float32)

    return actions, observations

def behavioural_cloning_training(model, actions, observations, lr, num_epochs, batch_size):

    # Convert observations and actions to tensors
    expert_actions = torch.tensor(actions, dtype=torch.float32, requires_grad=True)

    expert_observations = torch.tensor(observations, dtype=torch.float32, requires_grad=True)
    expert_observations = expert_observations.permute(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]
    expert_observations = expert_observations.float()

    # Set model to training mode
    model.policy.set_training_mode(True)

    # Define an optimizer
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    # Set loss function: Cross Entropy Loss for discrete action spaces
    loss_fn = nn.CrossEntropyLoss()

    model = train_cnn(expert_observations, expert_actions, num_epochs, batch_size, model, loss_fn, optimizer)

    # for epoch in range(num_epochs):
    #     # Shuffle data at the start of each epoch
    #     permutation = np.random.permutation(len(expert_observations))
    #     expert_observations = expert_observations[permutation]
    #     expert_actions = expert_actions[permutation]
        
    #     for i in range(0, len(expert_observations), batch_size):
    #         batch_obs = expert_observations[i:i + batch_size]
    #         batch_actions = expert_actions[i:i + batch_size]

    #         # Forward pass: compute predicted actions by passing observations to the policy
    #         logits, _, _ = model.policy(batch_obs)

    #         # Calculate loss
    #         loss = loss_fn(logits.float(), batch_actions.float())

    #         # Backward pass: compute gradient of the loss with respect to model parameters
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)

    #         # Update model parameters
    #         optimizer.step()

    #     print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

    return model


def behavioural_cloning(model, filepath, model_path, lr=1e-3, num_epochs=10, batch_size=64):

    actions, observations = load_data(filepath)

    model = behavioural_cloning_training(model, actions, observations, lr, num_epochs, batch_size)

    model.save(model_path)

    return model

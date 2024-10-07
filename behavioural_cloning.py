import torch
import torch.nn as nn
import numpy as np
import pickle
# from skimage.metrics import structural_similarity as ssim


def pretrain_ppo_with_bc(model, actions, observations, lr, num_epochs, batch_size, device='cpu'):
    model.policy.train()  # Switch to training mode
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    # Expert data should not require gradients
    expert_actions = torch.tensor(actions, dtype=torch.float32).to(device)
    expert_observations = torch.tensor(observations, dtype=torch.float32).to(device)

    dataset_size = len(observations)

    for epoch in range(num_epochs):
        # Shuffle data at the start of each epoch
        permutation = np.random.permutation(len(expert_observations)) 
        expert_obs = expert_observations[permutation]
        expert_acts = expert_actions[permutation]

        epoch_loss = 0.0
        num_batches = int(np.ceil(dataset_size / batch_size))

        # Mini-batch training
        for i in range(num_batches):
            batch_indices = np.random.choice(dataset_size, batch_size)
            obs_batch = expert_obs[batch_indices]
            action_batch = expert_acts[batch_indices]

            optimizer.zero_grad()

            # Forward pass: predict actions from observations
            predicted_actions, _, _ = model.policy(obs_batch)
            predicted_actions = torch.tensor(predicted_actions, dtype=torch.float32, requires_grad=True).to(device)
            predicted_actions = predicted_actions.squeeze()

            # Compute behavior cloning loss
            loss = loss_fn(predicted_actions, action_batch)
            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")

    return model

def pretrain_dqn_with_bc(model, actions, observations, lr, num_epochs, batch_size, device='cpu'):
    model.policy.train()  # Switch to training mode
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    # Expert data should not require gradients
    expert_actions = torch.tensor(actions, dtype=torch.float32).to(device)
    expert_observations = torch.tensor(observations, dtype=torch.float32).to(device)

    dataset_size = len(observations)

    for epoch in range(num_epochs):
        # Shuffle data at the start of each epoch
        permutation = np.random.permutation(len(expert_observations)) 
        expert_obs = expert_observations[permutation]
        expert_acts = expert_actions[permutation]

        epoch_loss = 0.0
        num_batches = int(np.ceil(dataset_size / batch_size))

        # Mini-batch training
        for i in range(num_batches):
            batch_indices = np.random.choice(dataset_size, batch_size)
            obs_batch = expert_obs[batch_indices]
            action_batch = expert_acts[batch_indices]

            optimizer.zero_grad()

            # Forward pass: predict actions from observations
            predicted_actions = model.policy(obs_batch)
            predicted_actions = torch.tensor(predicted_actions, dtype=torch.float32, requires_grad=True).to(device)
            predicted_actions = predicted_actions.squeeze()

            # Compute behavior cloning loss
            loss = loss_fn(predicted_actions, action_batch)
            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")

    return model

def pretrain_sac_with_bc(model, actions, observations, lr, num_epochs, batch_size, device='cpu'):
    model.actor.train()  # Switch to training mode
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    loss_fn = nn.MSELoss()

    # Expert data should not require gradients
    expert_actions = torch.tensor(actions, dtype=torch.float32).to(device)
    expert_observations = torch.tensor(observations, dtype=torch.float32).to(device)

    dataset_size = len(observations)

    for epoch in range(num_epochs):
        # Shuffle data at the start of each epoch
        permutation = np.random.permutation(len(expert_observations)) 
        expert_obs = expert_observations[permutation]
        expert_acts = expert_actions[permutation]

        epoch_loss = 0.0
        num_batches = int(np.ceil(dataset_size / batch_size))

        # Mini-batch training
        for i in range(num_batches):
            batch_indices = np.random.choice(dataset_size, batch_size)
            obs_batch = expert_obs[batch_indices]
            action_batch = expert_acts[batch_indices]

            optimizer.zero_grad()

            # Forward pass: predict actions from observations
            predicted_actions = model.actor(obs_batch)
            predicted_actions = torch.tensor(predicted_actions, dtype=torch.float32, requires_grad=True).to(device)
            predicted_actions = predicted_actions.squeeze()
            
            # Compute behavior cloning loss
            loss = loss_fn(predicted_actions, action_batch)
            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}")

    return model

def load_data(filepath, n_stack):
    # load in data
    with open(filepath, 'rb') as file:
        trajectories = pickle.load(file)

    trajectories = np.array(trajectories, dtype=object)

    # [obs, act, done, next_obs]

    observations = []

    stacked_obs = []

    # expert_observations = expert_observations.permute(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]
    
    for i, trajectory in enumerate(trajectories):

        stacked_obs.append(trajectory[0])

        if len(stacked_obs) == n_stack:
            # add stacked_obs to observations
            stacked_obs_arr = np.array(stacked_obs)
            stacked_obs_arr = stacked_obs_arr.reshape(n_stack, 84, 84)

            observations.append(stacked_obs_arr)

            if trajectory[2]:
                # end of attempt
                stacked_obs = []
            else:
                # remove first bit of stacked_obs
                stacked_obs.pop(0)

    # for some reason have to do it this way, otherwise it gets upset
    # observations = [np.array(obs, dtype=np.float32) for obs in trajectories[:, 0]]
    observations = np.array(observations)
    print(observations.shape)
    # observations = preprocess_observations(observations)

    actions = np.array(trajectories[:, 1], dtype=np.float32)

    return actions, observations

# def behavioural_cloning_training(model, actions, observations, lr, num_epochs, batch_size):

#     # Convert observations and actions to tensors
#     expert_actions = torch.tensor(actions, dtype=torch.float32, requires_grad=True)

#     expert_observations = torch.tensor(observations, dtype=torch.float32, requires_grad=True)
#     expert_observations = expert_observations.permute(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]
#     expert_observations = expert_observations.float()

#     # Set model to training mode
#     model.policy.set_training_mode(True)

#     # Define an optimizer
#     optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

#     # Set loss function: Cross Entropy Loss for discrete action spaces
#     loss_fn = nn.CrossEntropyLoss()

#     model = train_cnn(expert_observations, expert_actions, num_epochs, batch_size, model, loss_fn, optimizer)

#     return model


def behavioural_cloning(model_name, model, filepath, model_path, lr=1e-2, num_epochs=100, batch_size=64, n_stack=1):

    actions, observations = load_data(filepath, n_stack)

    if model_name == "PPO":
        print("PPO behaviour cloning starting")
        model = pretrain_ppo_with_bc(model, actions, observations, lr, num_epochs, batch_size)
    elif model_name == "DQN":
        print("DQN behaviour cloning starting")
        model = pretrain_dqn_with_bc(model, actions, observations, lr, num_epochs, batch_size)
    elif model_name == "SAC":
        print("SAC behaviour cloning starting")
        model = pretrain_sac_with_bc(model, actions, observations, lr, num_epochs, batch_size)

    print(f"behaviour cloning finished, being saved to {model_path}")
    model.save(model_path)

    return model


# def idk(nonexpert_pair, expert_acts, all_expert_obs):

#     nonexpert_act, nonexpert_obs = nonexpert_pair
#     some_threshold = 0.8

#     for i, expert_obs in enumerate(all_expert_obs):

#         score, diff = ssim(nonexpert_obs, expert_obs, full=True, channel_axis=2)
#         # compare two images by computing the Structural Similarity Index (SSIM)
#         # gives a score between -1 and 1, where 1 indicates identical images.

#         if score == 1 or score > some_threshold:
#             # so the obs is similar to an expert obs
#             is_similar = True
#             # now we check if the action is the same
#             # print("similar")
#             expert_act = expert_acts[i]
#             if expert_act == nonexpert_act:
#                 # could also check if it's similar? i.e. like right w/o shift kinda thing??
#                 return nonexpert_act, nonexpert_obs, 1
#             else:
#                 new_act = np.ones(13)
#                 proportion = 1/(13 - 1)

#                 new_act = new_act*proportion # set all other actions to be equally likely

#                 # index = np.nonzero(nonexpert_act)
#                 # set index of the nonexpert action to 0, as we don't want to do this action??
#                 new_act[int(nonexpert_act)] = 0

#                 return new_act, nonexpert_obs, 2

#     # if is_similar:
#     #     non_zero_indices = torch.nonzero(new_act)
#     #     if len(non_zero_indices) != 0:
#     #         proportion = 1/len(non_zero_indices)
#     #     else:
#     #         # lots of conflicting info from expert obs so we'll just say everything is equally likely??
#     #         proportion = 1/len(new_act)
#     #         new_act = torch.ones_like(new_act)

#     #     nonexpert_act = new_act*proportion # set all other actions to be equally likely

#     return nonexpert_act, nonexpert_obs, 0


# def main():
#     # load in the two trajectories datasets

#     results = []

#     nonexpert_acts, nonexpert_obs = load_data("/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/user_data_processed_for_bc/nonexpert_distance_bc_data.obj")
#     expert_acts, expert_obs = load_data("/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/user_data_processed_for_bc/expert_distance_bc_data.obj")

#     expert_acts = expert_acts[:10000]
#     expert_obs = expert_obs[:10000]

#     for i, obs in enumerate(nonexpert_obs[:10000]):
#         # print(obs.shape)
#         pair = (nonexpert_acts[i], obs)
#         _, _, indicator = idk(pair, expert_acts, expert_obs)

#         results.append(indicator)

#     # Create an empty dictionary to store counts
#     counts = {}

#     # Iterate through the list and count occurrences
#     for number in results:
#         if number in counts:
#             counts[number] += 1
#         else:
#             counts[number] = 1

#     print(counts)

# # if __name__=="__main__":
#     # main()

import pickle
import numpy as np
import copy

FILEPATH = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/bc_datasets/"

def load_data(training_data_name, level):

    full_filepath = f"{FILEPATH}{level}_{training_data_name}.pkl"

    with open(full_filepath, 'rb') as file:
        loaded_data = pickle.load(file)

    actions, observations = zip(*loaded_data)

    print(f"Loading {level} demo data from {full_filepath}.")

    observations = np.array(observations)
    # print(observations.shape)
    observations = observations.reshape(observations.shape[0], observations.shape[1], observations.shape[2])
    print(observations.shape)

    actions = np.array(actions)

    return observations, actions

def invert_actions(observations, actions):

    print(observations.shape)
    print(actions.shape)

    updated_observations = []
    updated_actions = []

    for observation, action in zip(observations, actions):
        action_probs = np.ones(13)
        action_probs = action_probs/12 # equal probability will pick the other 12 options
        action_probs[int(action)] = 0

        # for alt_action in range(13):
        #     if alt_action != action:
        updated_observations.append(observation)
        updated_actions.append(action_probs)

    updated_observations = np.array(updated_observations)
    updated_actions = np.array(updated_actions)

    print(updated_observations.shape)
    print(updated_actions.shape)

    return updated_observations, updated_actions

def swap_to_probs(observations, actions):

    updated_observations = []
    updated_actions = []

    for observation, action in zip(observations, actions):
        action_probs = np.zeros(13)
        action_probs[int(action)] = 1

        updated_observations.append(observation)
        updated_actions.append(action_probs)

    updated_observations = np.array(updated_observations)
    updated_actions = np.array(updated_actions)

    print(updated_observations.shape)
    print(updated_actions.shape)

    return updated_observations, updated_actions

def save_data(states, actions, level, filename):
    if states.shape[0] != actions.shape[0]:
        print(f"Something has gone wrong... :(")

    combined = zip(actions, states)

    filepath = f"{FILEPATH}{level}_{filename}"
    with open(f"{filepath}.pkl", "wb") as file:
        pickle.dump(combined, file)
        print(f"Dataset saved to: {filepath}.pkl")

def generate_new_dataset(level):

    nonexpert_observations, nonexpert_actions = load_data("nonexpert_classifier", level)

    expert_observations, expert_actions = load_data("expert_classifier", level)

    flipped_nonexpert_observations, flipped_nonexpert_actions = invert_actions(nonexpert_observations, nonexpert_actions)
    expert_observations, expert_actions = swap_to_probs(expert_observations, expert_actions)

    combined_observations = np.concatenate((flipped_nonexpert_observations, expert_observations))
    combined_actions = np.concatenate((flipped_nonexpert_actions, expert_actions))

    save_data(combined_observations, combined_actions, level, "amalgam_with_flipped_actions")


for level in ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]:
    print(level)
    generate_new_dataset(level)

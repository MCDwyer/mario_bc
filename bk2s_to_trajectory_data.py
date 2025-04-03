import retro
import numpy as np
import copy
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import pickle
from gymnasium.envs.registration import register
from GymEnvs.retro_env_wrapper import ReplayMarioEnv
from scipy.stats import entropy
from scipy.spatial.distance import directed_hausdorff
import itertools
from joblib import Parallel, delayed

register(
    id='MarioEnv-v0',
    entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
)

MAX_X = 3840
MAX_Y = 768
MARIO_X = 13
MARIO_Y = 16

MAX_SCORE = 1000
MAX_DISTANCE = 3840

DISTANCE_THRESHOLD = MAX_X/2

COLOUR_SCHEME = px.colors.qualitative.Plotly

FORCE_RELOAD = False
GENERATE_DATASETS = False
PLOTS = True
PLOT_HEATMAPS = True
MAKE_TABLE = False
FORCE_RELOAD_AGENTS = True

def process_observation(obs):
    # Convert the frame to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # Resize the frame to 84x84
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # Add a channel dimension
    obs = np.expand_dims(obs, axis=-1)

    # obs = np.transpose(obs, (0, 3, 1, 2))
    return obs

def generate_offline_rl_dataset(dir, filename, level, all_states, all_actions, all_next_states, all_dist_rewards, all_score_rewards):

    stacked_dones = np.zeros(len(all_states[0]), dtype=bool)
    stacked_dones[-1] = True
    stacked_states = np.array(all_states[0])

    for state_set in all_states[1:]:
        done_set = np.zeros(len(state_set), dtype=bool)
        done_set[-1] = True
        stacked_dones = np.concatenate((stacked_dones, done_set))

        state_set = np.array(state_set)
        stacked_states = np.concatenate((stacked_states, state_set))

    print(stacked_dones.shape)
    print(stacked_states.shape)

    stacked_next_states = np.array(all_next_states[0])

    for state_set in all_next_states[1:]:
        state_set = np.array(state_set)
        stacked_next_states = np.concatenate((stacked_next_states, state_set))

    print(stacked_next_states.shape)

    stacked_actions = np.array(all_actions[0])
    print(stacked_actions.shape)

    for action_set in all_actions[1:]:
        action_set = np.array(action_set)
        stacked_actions = np.concatenate((stacked_actions, action_set))

    stacked_dist_rewards = np.array(all_dist_rewards[0])
    print(stacked_dist_rewards.shape)

    for reward_set in all_dist_rewards[1:]:
        reward_set = np.array(reward_set)
        stacked_dist_rewards = np.concatenate((stacked_dist_rewards, reward_set))

    stacked_score_rewards = np.array(all_score_rewards[0])
    print(stacked_score_rewards.shape)

    for reward_set in all_score_rewards[1:]:
        reward_set = np.array(reward_set)
        stacked_score_rewards = np.concatenate((stacked_score_rewards, reward_set))

    # ((score_reward/MAX_SCORE)/2 + (dist_reward/MAX_DISTANCE)/2)*MAX_DISTANCE 
    stacked_combined_rewards = np.divide(stacked_score_rewards, (2*MAX_SCORE)) + np.divide(stacked_dist_rewards, (2*MAX_DISTANCE))
    stacked_combined_rewards = np.multiply(stacked_combined_rewards, MAX_DISTANCE)

    # distance dataset
    # ((obs, action, reward, next_obs, done))
    combined = zip(stacked_states, stacked_actions, stacked_dist_rewards, stacked_next_states, stacked_dones)

    filepath = f"{dir}{level}_{filename}_offline_dist"
    with open(f"{filepath}.pkl", "wb") as file:
        pickle.dump(combined, file)
        print(f"Dataset saved to: {filepath}.pkl")
    
    # score dataset
    combined = zip(stacked_states, stacked_actions, stacked_score_rewards, stacked_next_states, stacked_dones)

    filepath = f"{dir}{level}_{filename}_offline_score"
    with open(f"{filepath}.pkl", "wb") as file:
        pickle.dump(combined, file)
        print(f"Dataset saved to: {filepath}.pkl")

    # combined dataset
    combined = zip(stacked_states, stacked_actions, stacked_combined_rewards, stacked_next_states, stacked_dones)

    filepath = f"{dir}{level}_{filename}_offline_combined"
    with open(f"{filepath}.pkl", "wb") as file:
        pickle.dump(combined, file)
        print(f"Dataset saved to: {filepath}.pkl")

    return 

def generate_dataset_from_state_actions(dir, filename, level, all_states, all_actions):

    # combine the states and actions together?
    stacked_states = np.array(all_states[0])
    print(stacked_states.shape)

    for state_set in all_states[1:]:
        state_set = np.array(state_set)
        # state_set = state_set.reshape(state_set.shape[3], state_set.shape[0], state_set.shape[1], state_set.shape[2])
        stacked_states = np.concatenate((stacked_states, state_set))

    stacked_actions = np.array(all_actions[0])
    # stacked_actions = stacked_actions.reshape(1, stacked_actions.shape[0])
    print(stacked_actions.shape)

    for action_set in all_actions[1:]:
        action_set = np.array(action_set)
        # action_set = action_set.reshape(1, action_set.shape[0])
        stacked_actions = np.concatenate((stacked_actions, action_set))

    print(stacked_states.shape)
    print(stacked_actions.shape)

    if stacked_states.shape[0] != stacked_actions.shape[0]:
        print(f"Something has gone wrong... :(")

    combined = zip(stacked_actions, stacked_states)

    filepath = f"{dir}{level}_{filename}"
    with open(f"{filepath}.pkl", "wb") as file:
        pickle.dump(combined, file)
        print(f"Dataset saved to: {filepath}.pkl")
    
    return

def extract_info_from_bk2s_test(bk2_file, level):
    try:
        movie = retro.Movie(bk2_file)
    except:
        print(bk2_file)
        movie = retro.Movie(bk2_file)

    movie.step()

    env = ReplayMarioEnv(game=movie.get_game(), state=level)
    env.initial_state = movie.get_state()

    prev_state = env.reset()

    trajectories = []
    action_distribution = np.zeros(13)
    actions = []
    states = []
    # reward = 0
    total_dist_reward = 0
    total_score_reward = 0
    total_combined_reward = 0

    step = 0
    max_score = 0

    prev_score = 0

    stacked_actions = []

    while movie.step():
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))

        stacked_actions.append(copy.deepcopy(keys))
        step += 1

        # if 'death_log' in info:
        #     if info['death_log']:
        #         if info["death_log"]['info']["player_state"] in {4, 6, 8, 11}:
        #             print(info['death_log'])

        # if step%4 == 0 or done: # only want every 4 timesteps to match agents
        if len(stacked_actions) == 4:
            # print(stacked_actions)
            obs, reward, done, _, info = env.step(stacked_actions)
            x = info["x_frame"]*256 + info["x_position_in_frame"]

            y = ((info["y_frame"]*256) + info["y_position_in_frame"])

            action = map_from_retro_action(keys)
            trajectories.append([x, y])
            processed_obs = process_observation(prev_state)
            states.append(processed_obs)
            actions.append(action)
            action_distribution[int(action)] += 1
            # next_states.append(process_observation(obs))
            # infos.append(info)

            dist_reward = env.dist_reward
            score_reward = env.score_reward
            combined_reward = env.combined_reward
            total_dist_reward += dist_reward
            total_score_reward += int(score_reward)
            total_combined_reward += combined_reward

            prev_state = obs
            prev_score = info["score"]

            if prev_score > max_score:
                max_score = prev_score

            if dist_reward < -2000:
                # print(reward, score_reward, dist_reward)
                print(f"Why isn't this working!!!! player state = {info['player_state']}")

            stacked_actions = []

    print(step)
    env.close()

    # check that there is actually a trajectory here:
    if trajectories:
        trajectories = np.array(trajectories)

        x_coords = trajectories[:, 0]
        y_coords = trajectories[:, 1]

        if len(set(x_coords)) == 1 or len(set(actions)) == 1:
            return None, None, None, None, None, None, None, None, None, None

        death_log = info["death_log"] if "death_log" in info else {}
        death_type = death_log["type"] if "type" in death_log else None

        return trajectories, states, actions, total_dist_reward, total_score_reward, total_combined_reward, action_distribution, max_score, death_type, death_log
    
    return None, None, None, None, None, None, None, None, None, None

def extract_info_from_bk2s(bk2_file, level):
    try:
        movie = retro.Movie(bk2_file)
    except:
        print(bk2_file)
        movie = retro.Movie(bk2_file)

    movie.step()

    env = retro.make(game=movie.get_game(), state=level)#, obs_type=retro.Observations.RAM)
    # env = ProcessedFrame(env)
    env.initial_state = movie.get_state()

    # print(movie.get_game())
    # print(env.initial_state)
    
    prev_state = env.reset()
    
    trajectories = []
    action_distribution = np.zeros(13)
    actions = []
    states = []
    next_states = []
    dist_rewards = []
    score_rewards = []
    # reward = 0
    total_dist_reward = 0
    total_score_reward = 0
    total_combined_reward = 0

    step = 0
    level = None
    max_score = 0

    prev_position = 40
    prev_score = 0

    state_change = False
    death_log = {}

    while movie.step():
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))

        obs, _, done, info = env.step(keys)
        step += 1

        # info = copy.deepcopy(info)

        if level is None:
            level = info["level"]

        if step%4 == 0: # only want every 4 timesteps?
            state_change = False

            # print(info["player_state"])
            last_info = copy.deepcopy(info)

            while info["player_state"] != 8:# and info["player_state"] != 11:
                # keep going as non playable bit?
                obs, _, done, info = env.step(keys)
                state_change = True

                if not death_log: # only append to death log once
                    if info["player_state"] == 11:
                        death_log = {"type": "enemy", "info": copy.deepcopy(last_info)}
                    elif info["time"] == last_info["time"]:
                        death_log = {"type": "fall", "info": copy.deepcopy(last_info)}
                    elif info["time"] == 0:
                        death_log = {"type": "timeout", "info": copy.deepcopy(last_info)}
                    elif info["player_state"] == 4:
                        death_log = {"type": "flagpole", "info": copy.deepcopy(info)}

                last_info = copy.deepcopy(info)

            x = info["x_frame"]*256 + info["x_position_in_frame"]

            y = ((info["y_frame"]*256) + info["y_position_in_frame"])

            if info["lives"] < 2 or info["level"] != level:
                break

            if state_change:
                dist_reward = 0
                score_reward = 0
                state_change = False
            else:
                dist_reward = (x - prev_position)
                score_reward = ((int(info['score'])*10) - prev_score)

            action = map_from_retro_action(keys)
            trajectories.append([x, y])
            processed_obs = process_observation(prev_state)
            states.append(processed_obs)
            processed_next_obs = process_observation(obs)
            next_states.append(processed_next_obs)
            actions.append(action)
            action_distribution[int(action)] += 1
            dist_rewards.append(dist_reward)
            score_rewards.append(score_reward)

            total_dist_reward += dist_reward
            total_score_reward += score_reward
            total_combined_reward += ((score_reward/MAX_SCORE)/2 + (dist_reward/MAX_DISTANCE)/2)*MAX_DISTANCE 

            prev_position = x
            prev_score = int(info['score'])*10
            prev_state = obs

            if prev_score > max_score:
                max_score = prev_score

            if dist_reward < -2000:
                # print(reward, score_reward, dist_reward)
                print(f"Why isn't this working!!!! player state = {info['player_state']}")

    env.close()

    # check that there is actually a trajectory here:
    if trajectories:
        trajectories = np.array(trajectories)

        x_coords = trajectories[:, 0]
        y_coords = trajectories[:, 1]

        if len(set(x_coords)) == 1 or len(set(actions)) == 1:
            return None, None, None, None, None, None, None, None, None, None, None, None, None

        death_type = death_log["type"] if "type" in death_log else None

        return trajectories, states, actions, next_states, dist_rewards, score_rewards, total_dist_reward, total_score_reward, total_combined_reward, action_distribution, max_score, death_type, death_log
    
    return None, None, None, None, None, None, None, None, None, None, None, None, None

def map_from_retro_action(binary_action):
    # this function maps from the retro env action space (binary_action) back to the discrete action space
    SHIFT_INDEX = 0 
    LEFT_INDEX = 6
    RIGHT_INDEX = 7
    UP_INDEX = 4
    DOWN_INDEX = 5
    JUMP_INDEX = 8
    NO_ACTION = 12

    # Reverse action mapping based on the given map_to_retro_action function
    reverse_action_mapping = {
        (UP_INDEX,): 0,
        (DOWN_INDEX,): 1,
        (LEFT_INDEX,): 2,
        (RIGHT_INDEX,): 3,
        (JUMP_INDEX,): 4,
        (SHIFT_INDEX,): 5,
        # 6, 8
        (LEFT_INDEX, JUMP_INDEX): 6,
        # 7, 8
        (RIGHT_INDEX, JUMP_INDEX): 7,
        # 0, 6
        (SHIFT_INDEX, LEFT_INDEX): 8,
        # 0, 7
        (SHIFT_INDEX, RIGHT_INDEX): 9,
        # 0, 6, 8
        (SHIFT_INDEX, LEFT_INDEX, JUMP_INDEX): 10,
        # 0 7, 8
        (SHIFT_INDEX, RIGHT_INDEX, JUMP_INDEX): 11,
        (): NO_ACTION
    }

    # Get the indices where the binary_action is True
    active_indices = tuple(index for index, active in enumerate(binary_action) if active)

    sort_indices = tuple(sorted(active_indices))

    # Return the corresponding discrete action
    return reverse_action_mapping.get(sort_indices, NO_ACTION)

def load_in_demo_data(demo_dir, level):
    # return trajectories, actions, reward, action_distribution
    demo_dir_level = f"{demo_dir}/{level}"
    
    full_file_list = os.listdir(demo_dir_level)

    all_trajectories = []
    all_actions = []
    all_states = []
    all_next_states = []
    all_individual_dist_rewards = []
    all_individual_score_rewards = []
    all_dist_rewards = []
    all_score_rewards = []
    all_combined_rewards = []
    all_action_distributions = []
    all_max_scores = []
    all_death_types = {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0}
    all_death_logs = []
    # combined_action_distribution = np.zeros(13)

    for filename in full_file_list:
        if filename.endswith(".bk2"):
            full_filename = f"{demo_dir_level}/{filename}"
            trajectories, states, actions, next_states, dist_rewards, score_rewards, total_dist_reward, total_score_reward, total_combined_reward, action_distribution, max_score, death_type, death_log = extract_info_from_bk2s(full_filename, level)

            if trajectories is not None:
                all_trajectories.append(trajectories)
                all_actions.append(actions)
                all_states.append(states)
                all_next_states.append(next_states)
                all_individual_dist_rewards.append(dist_rewards)
                all_individual_score_rewards.append(score_rewards)
                all_dist_rewards.append(total_dist_reward)
                all_score_rewards.append(total_score_reward)
                all_combined_rewards.append(total_combined_reward)
                all_action_distributions.append(action_distribution)
                all_max_scores.append(max_score)
                if death_type is not None:
                    all_death_types[death_type] += 1
                all_death_logs.append(death_log)
                # combined_action_distribution += action_distribution

    return all_trajectories, all_states, all_actions, all_next_states, all_individual_dist_rewards, all_individual_score_rewards, all_dist_rewards, all_score_rewards, all_combined_rewards, all_action_distributions, all_max_scores, all_death_types, all_death_logs#, combined_action_distribution

def load_in_agent_data(agent_dir, level, check_for=""):
    full_file_list = filter(
        lambda f: os.path.isdir(os.path.join(agent_dir, f)),  # Use full path
        os.listdir(agent_dir)  # List only filenames
    )

    # Convert filter object to a list
    full_file_list = list(full_file_list)

    agents = {}
    
    for directory in full_file_list:
        # agent type

        if "bc_only" in directory:
            agent_type = directory[:-len("_0_bc_only")]
            agent_type += "_bc_only"
        else:
            agent_type = directory[:-2] # remove index and _

        if agent_type not in agents:
            agents[agent_type] = {"trajectories": [],
                                "actions": [],
                                "dist_rewards": [],
                                "score_rewards": [],
                                "combined_rewards": [],
                                "action_distributions": [],
                                "max_score": [],
                                "fall_ends": 0,
                                "enemy_ends": 0,
                                "timeout_ends": 0,
                                "flagpole_ends": 0,
                                # "death_types": {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0},
                                "death_logs": [],
                                }

        full_agent_dir = f"{agent_dir}/{directory}"
        dataframes = os.listdir(full_agent_dir)

        for dataframe in dataframes:
            if level in dataframe:
                full_path = f"{full_agent_dir}/{dataframe}"
                df = pd.read_pickle(full_path)
                
                df = df.loc[:, ~df.applymap(lambda x: isinstance(x, list) and len(x) == 0).all()]
        
                df_dict = df.to_dict()
                # print(df_dict.keys())

                if "dist_rewards" not in list(df_dict.keys()):
                    print(full_path)
                else:
                    for df_key in agents[agent_type]:
                        if df_key.endswith("ends"):
                            value = int(df_dict[df_key][0]) # pretty sure that it's duplicating the value for each trial for some reason, so only need one of them (could be any of them as they;re all the same value)
                            agents[agent_type][df_key] = value + agents[agent_type][df_key]#[key] for key in agents[agent_type][df_key]}
                        else:                                                         
                            df_list = [value for _, value in sorted(df_dict[df_key].items())]
                            agents[agent_type][df_key] = agents[agent_type][df_key] + df_list
        
    return agents

def get_trajectory_metrics(trajectory, infos=None):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    # time = 400 - np.min(infos["time"])

    # high_score = np.max(infos["score"])
    # time_taken = time

    furthest_distance = np.max(x)
    highest_distance = np.max(y)
    average_height = np.mean(y)
        
    # total_distance = 0
    # for k in range(1, len(x)):
    #     total_distance += abs(x[k] - x[k - 1])
    # average_speed = total_distance/time

    return furthest_distance, highest_distance, average_height#, average_speed, high_score, time

def distance_expert(trajectory):

    x = trajectory[:, 0]

    return np.max(x) > DISTANCE_THRESHOLD

def split_by_indices(all_data, indices):
    split_1 = []
    split_2 = []

    for i, data in enumerate(all_data):
        if i in indices:
            split_1.append(copy.deepcopy(data))
        else:
            split_2.append(copy.deepcopy(data))

    return split_1, split_2

def dataframe_saving(df, filename):

    if not df.empty:
        print(f"{filename} being saved with {len(df.index)} rows of data.")
        df.to_pickle(filename) 
    else:
        print(f"{filename} not saved as dataframe is empty.")

def compute_statistics(df):
    results = {}

    for key in df.columns:  # Iterate over "amalgam demo data", "nonexpert demo data", etc.
        print(f"Processing: {key}")
        # Extract rewards (handle both list & dict cases)
        results[key] = {}

        total_number_of_attempts = 0

        for reward_type in ["dist_rewards", "score_rewards", "combined_rewards", "max_score"]:
            rewards_data = df.loc[reward_type, key]
            if isinstance(rewards_data, dict):  # If rewards_data is a dict, extract values
                rewards = np.array(list(rewards_data.values()))
            elif isinstance(rewards_data, list):  # If already a list, convert to NumPy array
                rewards = np.array(rewards_data)
            else:
                raise TypeError(f"Unexpected type {type(rewards_data)} for {reward_type} in {key}")

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            results[key][f"mean_{reward_type}"] = mean_reward
            results[key][f"std_{reward_type}"] = std_reward
            total_number_of_attempts = len(rewards)
            
        # Extract action distributions dictionary
        action_dist_data = df.loc["action_distributions", key]
        if isinstance(action_dist_data, dict):
            combined_action_dist = np.sum(np.array(list(action_dist_data.values())), axis=0)  # Sum across episodes
        elif isinstance(rewards_data, list):  # If already a list, convert to NumPy array
            combined_action_dist = np.sum(np.array(action_dist_data), axis=0)
        else:
            raise TypeError(f"Unexpected type {type(action_dist_data)} for action distributions in {key}")

        results[key]["combined_action_distribution"] = combined_action_dist

        results[key]["end_reasons_percentage"] = {"fall": 0.0, "enemy": 0.0, "flagpole": 0.0, "timeout": 0.0}

        for end_type in ["fall_ends", "enemy_ends", "timeout_ends", "flagpole_ends"]:
            end_data = df.loc[end_type, key]
            if isinstance(end_data, dict):  # If rewards_data is a dict, extract values
                end_data = np.array(list(end_data.values()))
            elif isinstance(end_data, list):  # If already a list, convert to NumPy array
                end_data = np.array(end_data)
            end_type_key = end_type.split("_")[0]
            results[key]["end_reasons_percentage"][end_type_key] = np.sum(end_data)/total_number_of_attempts

        traj_data = df.loc["trajectories", key]
        if isinstance(traj_data, dict):  # If rewards_data is a dict, extract values
            traj_data = np.array(list(traj_data.values()))
        elif isinstance(traj_data, list):  # If already a list, convert to NumPy array
            traj_data = np.array(traj_data)
        
        max_horizontal = []
        max_vertical = []
        
        for trajectory in traj_data:
            trajectory = np.array(trajectory)
            max_horizontal.append(np.max(trajectory[:, 0]))
            max_vertical.append(np.max(trajectory[:, 1]))

        results[key][f"mean_max_horizontal_position"] = np.mean(np.array(max_horizontal))
        results[key][f"std_max_horizontal_position"] = np.std(np.array(max_horizontal))
        results[key][f"mean_max_vertical_position"] = np.mean(np.array(max_vertical))
        results[key][f"std_max_vertical_position"] = np.std(np.array(max_vertical))

        time_taken = []
        final_positions = []

        death_logs = df.loc["death_logs", key]
        if isinstance(death_logs, dict):  # If rewards_data is a dict, extract values
            death_logs = np.array(list(death_logs.values()))
        elif isinstance(death_logs, list):  # If already a list, convert to NumPy array
            death_logs = np.array(death_logs)
        
        for log in death_logs:
            if "info" in log:
                info = log["info"]
                x = info["x_frame"]*256 + info["x_position_in_frame"]
                y = ((info["y_frame"]*256) + info["y_position_in_frame"])

                final_positions.append([x, y])
                time_taken.append((400 - info["time"]))

        results[key]["end_positions"] = final_positions
        results[key][f"mean_time_taken"] = np.mean(time_taken)
        results[key][f"std_time_taken"] = np.std(time_taken)

    return results

def plot_patterns(names_list):
    colours = []
    pattern_shape_sequence = []
    marker_shapes = []

    for agent_type in names_list:
        if "demo" in agent_type.lower():
            pattern_shape_sequence.append("/")
            marker_shapes.append("diamond")
        elif "bc_only" in agent_type.lower():
            pattern_shape_sequence.append("+")
            marker_shapes.append("cross")
        elif "unsupervised" in agent_type.lower():
            pattern_shape_sequence.append("")
            marker_shapes.append("circle")
        else:
            pattern_shape_sequence.append("x")
            marker_shapes.append("x")

        if "best" in agent_type.lower():
            marker_shapes[-1] = "star"

        if "nonexpert" in agent_type.lower():
            colours.append(COLOUR_SCHEME[0])
        elif "expert" in agent_type.lower():
            colours.append(COLOUR_SCHEME[1])
        elif "amalgam" in agent_type.lower():
            colours.append(COLOUR_SCHEME[2])
        else:
            colours.append(COLOUR_SCHEME[3])

    return colours, pattern_shape_sequence, marker_shapes

def mean_plots(df, level, with_errors=True, plot_dir=""):
    all_rewards_types = []
    other_figs = []

    for row_name in df.index.tolist():
        if "mean" in row_name:
            if "rewards" in row_name:
                clean_row_name = row_name[len("mean_"):]
                all_rewards_types.append(clean_row_name)
            else:
                clean_row_name = row_name[len("mean_"):]
                other_figs.append(clean_row_name)

    _mean_plots(df, level, with_errors, plot_dir, TRAINING_REWARD, "Reward", all_rewards_types)
    _mean_plots(df, level, with_errors, plot_dir, TRAINING_REWARD, "Other Metrics", other_figs, individual_figs=False)

def _mean_plots(df, level, with_errors, plot_dir, training_reward, name, row_list, individual_figs=True):

    sig_figs = 3
    # all_rewards_types = []
    # other_figs = []

    # for row_name in df.index.tolist():
    #     if "mean" in row_name:
    #         if "rewards" in row_name:
    #             clean_row_name = row_name[len("mean_"):]
    #             all_rewards_types.append(clean_row_name)
    #         else:
    #             clean_row_name = row_name[len("mean_"):]
    #             other_figs.append(clean_row_name)

    specs = [[{}] for _ in range(len(row_list))]

    subplot_fig = make_subplots(len(row_list), 1, subplot_titles=row_list, shared_xaxes=True, specs = specs,
                          vertical_spacing = 0.05)

    colours = []
    pattern_shape_sequence = []

    colours, pattern_shape_sequence, _ = plot_patterns(df.columns.tolist())

    for i, reward_type in enumerate(row_list):
        # Extract mean and std deviation values
        mean_rewards = df.loc[f"mean_{reward_type}"]
        std_rewards = df.loc[f"std_{reward_type}"]

        # Create a DataFrame for Plotly
        plot_df = pd.DataFrame({
            "Agent Type": mean_rewards.index,  # Column names as agent types
            "Mean": mean_rewards.values,
            "Std": std_rewards.values
        })

        if with_errors:
            # Create bar plot with error bars
            fig = px.bar(
                plot_df,
                x="Agent Type",
                y="Mean",
                error_y="Std",
                title=f"Mean {name} ({reward_type}) with Standard Deviation per Agent for Level: {level}",
                labels={"Mean": "Mean", "Agent Type": "Agent Type"},
                color="Agent Type",
                color_discrete_sequence = colours,
                pattern_shape="Agent Type",
                pattern_shape_sequence=pattern_shape_sequence,
                text_auto=f'.{sig_figs}s'
            )

            subplot_fig.add_trace(
                go.Bar(
                    x=plot_df["Agent Type"],
                    y=plot_df["Mean"],
                    error_y=dict(type='data', array=plot_df["Std"], visible=True),
                    # name=reward_type,
                    marker=dict(
                        color=colours,
                        pattern_shape=pattern_shape_sequence,  # Different patterns per bar
                        pattern_fgcolor="grey"
                    ),
                    text=[f"{val:.{sig_figs}g}" for val in plot_df["Mean"]],
                    textposition='outside',
                    showlegend=False),
                    row=(i+1),
                    col=1)

            plot_path = f"{level}_mean_{name}_{reward_type}_with_errors"
        else:
            # Create bar plot with error bars
            fig = px.bar(
                plot_df,
                x="Agent Type",
                y="Mean",
                title=f"Mean {name} ({reward_type}) per Agent for Level: {level}",
                labels={"Mean": "Mean", "Agent Type": "Agent Type"},
                color="Agent Type",
                color_discrete_sequence = colours,
                pattern_shape="Agent Type",
                pattern_shape_sequence=pattern_shape_sequence,
                text_auto=f'.{sig_figs}s'
            )

            subplot_fig.add_trace(
                go.Bar(
                    x=plot_df["Agent Type"],
                    y=plot_df["Mean"],
                    # name=reward_type,
                    marker=dict(
                        color=px.colors.qualitative.Pastel[:len(df.columns.tolist())],
                        pattern_shape=pattern_shape_sequence,  # Different patterns per bar
                        pattern_fgcolor="grey"
                    ),
                    text=[f"{val:.{sig_figs}g}" for val in plot_df["Mean"]],
                    textposition="outside",
                    showlegend=False),
                    row=(i+1),
                    col=1)

            plot_path = f"{level}_mean_{name}_{reward_type}"

        if individual_figs:
            fig.update_layout(showlegend=False)#, textposition="outside", cliponaxis=False)
            plotly_save_with_dir_check(fig, plot_dir, plot_path)

    plot_path = f"{level}_mean_{name}"
    plot_path += "_with_errors" if with_errors else ""
    plot_path += "_subplot"

    subplot_fig.update_layout(
        margin=dict(t=50, b=40, l=30, r=10),  # Reduce top margin (t), adjust others as needed
        height=800,  # Increase figure height
        width=800,   # Increase figure width
        title_text=f"Mean {name} for Level: {level} - Trained with {training_reward} reward",
    )

    plotly_save_with_dir_check(subplot_fig, plot_dir, plot_path)

def plot_action_distributions(df, level, plot_dir=""):
    # Extract action distributions
    action_distributions = df.loc["combined_action_distribution"]

    action_distr = []
    for i, action_distribution in enumerate(action_distributions):
        # print(action_distribution)
        # print(np.sum(action_distribution))
        action_distr.append(action_distribution/np.sum(action_distribution))

    fig = make_subplots(3, 4, subplot_titles=list(df.keys()))
    row = 1
    col = 1

    for i, agent_type in enumerate(action_distributions.index):
        action_distribution = action_distributions[i]

        fig.add_trace(
            go.Bar(
                x=list(range(13)), 
                y=action_distr[i],
                # name=agent_type
                showlegend=False
            ), row=row, col=col)

        col += 1

        if col > 4:
            col = 1
            row += 1


    fig.update_layout(title=f"Normalised Action Distributions per Agent on {level}", xaxis_title='Action Index', yaxis_title="Normalised Action Count")
    plot_path = f"{level}_action_distributions"

    plotly_save_with_dir_check(fig, plot_dir, plot_path)

# heatmap metrics
def heatmap_entropy(heatmap):
    flat = heatmap.flatten()
    p = flat / (np.sum(flat) + 1e-8)  # Normalize to probability
    return entropy(p, base=2)

def heatmap_variance(heatmap):
    return np.var(heatmap)

def heatmap_std(heatmap):
    return np.std(heatmap)

def top_k_concentration(heatmap, k=0.1):
    flat = heatmap.flatten()
    sorted_vals = np.sort(flat)[::-1]
    top_k = int(len(flat) * k)
    return np.sum(sorted_vals[:top_k]) / (np.sum(flat) + 1e-8)

def normalise_trajectory(traj, x_max, y_max):
    return [(x / x_max, y / y_max) for x, y in traj]

# Hausdorff: symmetric version
def hausdorff_distance(path1, path2):
    d1 = directed_hausdorff(path1, path2)[0]
    d2 = directed_hausdorff(path2, path1)[0]
    return max(d1, d2)

def remove_consecutive_duplicates(traj):
    not_list = False

    if type(traj) is np.ndarray:
        not_list = True
        traj = list(traj)

    if not traj:
        return []
    cleaned = [traj[0]]

    for pt in traj[1:]:
        if not_list:
            if not np.equal(pt, cleaned[-1]).all():
                cleaned.append(pt)
        else:
            if pt != cleaned[-1]:
                cleaned.append(pt)

    if not_list:
        cleaned = np.array(cleaned)

    return cleaned

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def discrete_frechet(P, Q):
    """
    Iterative version of discrete Fréchet distance between two paths P and Q.
    Each path is a list of 2D points: [(x, y), ...]
    """
    P = remove_consecutive_duplicates(P)
    Q = remove_consecutive_duplicates(Q)

    n, m = len(P), len(Q)
    ca = np.full((n, m), np.inf)

    ca[0, 0] = euclidean(P[0], Q[0])

    # First row
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], euclidean(P[0], Q[j]))

    # First column
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], euclidean(P[i], Q[0]))

    # Fill rest
    for i in range(1, n):
        for j in range(1, m):
            dist = euclidean(P[i], Q[j])
            min_prev = min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1])
            ca[i, j] = max(min_prev, dist)

    return ca[n - 1, m - 1]

def downsample(traj, step=5):
    return traj[::step] if len(traj) > step else traj

def pairwise_path_metrics(trajectories):
    # hausdorff_vals = []
    # # frechet_vals = []

    pairs = list(itertools.combinations(trajectories, 2))
    hausdorff_vals = Parallel(n_jobs=-1)(delayed(hausdorff_distance)(p1, p2) for p1, p2 in pairs)

    trajectories = [downsample(t, step=10) for t in trajectories]
    pairs = list(itertools.combinations(trajectories, 2))

    frechet_vals = Parallel(n_jobs=-1)(delayed(discrete_frechet)(p1, p2) for p1, p2 in pairs)

    return {
        "mean_hausdorff": np.mean(hausdorff_vals),
        "max_hausdorff": np.max(hausdorff_vals),
        "min_hausdorff": np.min(hausdorff_vals),
        "std_hausdorff": np.std(hausdorff_vals),
        "mean_frechet": np.mean(frechet_vals),
        "max_frechet": np.max(frechet_vals),
        "min_frechet": np.min(frechet_vals),
        "std_frechet": np.std(frechet_vals),
    }

def get_top_k_curve(heatmap, k_values=None):
    if k_values is None:
        k_values = np.linspace(0.00, 1, 1000)  # 1% to 100%

    flat = heatmap.flatten()
    sorted_vals = np.sort(flat)[::-1]
    total = np.sum(sorted_vals)

    if total == 0:
        return k_values, np.zeros_like(k_values)

    cumulative = np.cumsum(sorted_vals)
    curve = []

    for k in k_values:
        idx = int(len(sorted_vals) * k)
        idx = max(1, idx)
        curve.append(cumulative[idx - 1] / total)

    return k_values, curve

# end heatmap metrics
def get_mario_heatmap(trajectories, mario=True, level=None):

    if level is not None:
        num_screens = {"Level1-1": 14,
                        "Level2-1": 14,
                        "Level3-1": 14,
                        "Level4-1": 16,
                        "Level5-1": 15,
                        "Level6-1": 13,
                        "Level7-1": 15,
                        "Level8-1": 26}
        x_grid_max = int(256*num_screens[level]) + MARIO_X
    else:
        x_grid_max = MAX_X + MARIO_X #4000

    y_grid_max = MAX_Y + MARIO_Y #800

    heatmap = np.zeros(((y_grid_max + 1), (x_grid_max + 1)))
    print(heatmap.shape)

    for trajectory in trajectories:
        trajectory = remove_consecutive_duplicates(trajectory)
        trajectory_heatmap = np.zeros_like(heatmap)
        
        for state in trajectory:
            x, y = state
        
            if not mario:
                if x < x_grid_max and y < y_grid_max:
                    trajectory_heatmap[int(y), int(x)] += 1
            else:
                for i in range(MARIO_X):
                    new_x = x + i
                    for j in range(MARIO_Y):
                        new_y = y + j
                
                        if new_x < x_grid_max and new_y < y_grid_max:
                            trajectory_heatmap[int(new_y), int(new_x)] += 1
            
        trajectory_heatmap = np.clip(trajectory_heatmap, 0, 1)
        heatmap = heatmap + trajectory_heatmap

    heatmap = heatmap[::-1, :]

    # Compute metrics
    metrics = {
        "entropy": heatmap_entropy(heatmap),
        "variance": heatmap_variance(heatmap),
        "std_dev": heatmap_std(heatmap),
        "top_k_values": get_top_k_curve(heatmap)
    }

    # normalised_trajectories = [normalise_trajectory(traj, x_grid_max, y_grid_max) for traj in trajectories]

    # path_metrics = pairwise_path_metrics(normalised_trajectories)

    # metrics.update(path_metrics)

    return heatmap, x_grid_max, y_grid_max, metrics

def colorbar(n):
    return dict(
        tick0 = 0,
        title = "Log colour scale",
        tickmode = "array",
        tickvals = np.linspace(0, n, n+1),
        ticktext = 10**np.linspace(0, n, n+1))

def plot_mario_heatmap(trajectories, plot_path, level, info="", mario=True, max_value=None, colour_scale="Hot", x_axis_max=MAX_X, y_axis_max=MAX_Y):
    
    heatmap, x_grid_max, y_grid_max, metrics = get_mario_heatmap(trajectories, mario, level)
    
    n = int(np.round(np.log10(np.max(heatmap))))

    log_heatmap = np.log10(heatmap)
    log_heatmap = np.nan_to_num(log_heatmap)
    
    # heatmap = heatmap/np.sum(heatmap)
    heatmap = heatmap/len(trajectories)

    if max_value is None:
        max_value = np.max(heatmap)

    filepath_info = ""

    fig = go.Figure(data=go.Heatmap(
        z=heatmap,#[:, int(x_threshold)],
        x=np.array(range(x_grid_max)),#int(x_threshold))),
        y=np.array(range(y_grid_max)),
        zmin=0, 
        zmax=max_value,
        colorscale=colour_scale #'Viridis'
    ))

    fig.update_layout(
        title=f'{level} Heatmap - {info}',
        xaxis_range=[0, x_axis_max],
        yaxis_range=[0, y_axis_max],
    )

    if info:
        filepath_info = "_" + info.replace(" ", "_")

    plot_filename = f"{level}{filepath_info}_heatmap"
    plotly_save_with_dir_check(fig, plot_path, plot_filename)
    np.save(f"{plot_path}{plot_filename}", heatmap)

    fig = go.Figure(data=go.Heatmap(
        z=log_heatmap,#[:, int(x_threshold)],
        x=np.array(range(x_grid_max)),#int(x_threshold))),
        y=np.array(range(y_grid_max)),
        zmin=0, 
        zmax=np.max(log_heatmap),
        colorbar = colorbar(n+1),
        colorscale=colour_scale #'Viridis'
    ))

    fig.update_layout(
        title=f'{level} Heatmap - {info}',
        xaxis_range=[0, x_axis_max],
        yaxis_range=[0, y_axis_max],
    )

    plot_filename = f"{level}{filepath_info}_log_heatmap"
    plotly_save_with_dir_check(fig, plot_path, plot_filename)
    np.save(f"{plot_path}{plot_filename}", log_heatmap)

    return max_value, metrics


def plotly_save_with_dir_check(fig, plot_path, plot_filename):

    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    fig.write_image(f"{plot_path}{plot_filename}.png")
    fig.write_html(f"{plot_path}{plot_filename}.html")

    print(f"Figures saved to {plot_path}{plot_filename}(.png|.html).")


def plot_level_scatters(statistic_dict, plot_dir, agent_types, plot_error_bars=False):

    x_data = list(statistic_dict.keys())

    for reward_type in ["dist_rewards", "score_rewards", "combined_rewards"]:
        y_data = {}
        err_y_data = {}
        fig = go.Figure()
        for key in statistic_dict: # stats dictionary keys are levels

            for agent_type in statistic_dict[key].columns.tolist():
                if agent_type not in agent_types:
                    continue

                if agent_type not in y_data:
                    y_data[agent_type] = []
                    err_y_data[agent_type] = []

                y_value = statistic_dict[key].loc[f"mean_{reward_type}", agent_type]
                y_data[agent_type].append(y_value)
                err_y_value = statistic_dict[key].loc[f"std_{reward_type}", agent_type]
                # print(err_y_value)
                err_y_data[agent_type].append(err_y_value)

        colours, _, markers = plot_patterns(list(y_data.keys()))

        for i, agent_type in enumerate(y_data):

            fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data[agent_type],
                    mode='markers',
                    marker_symbol=markers[i],
                    error_y=dict(
                        type='data',
                        array=err_y_data[agent_type],
                        visible=plot_error_bars
                    ),
                    marker_color=colours[i],
                    name=agent_type
                ))

            fig.update_layout(
                title=f"Mean {reward_type} by level",
                xaxis_title="Level",
                yaxis_title=reward_type,
                # template="plotly_white",
                # xaxis_tickangle=-45,
                # yaxis=dict(
                #     range=[0, 1.05]  # Fixed y-axis range from 0 to 5
                # ),
            )

        filename = f"{reward_type}_per_level_scatter"
        plotly_save_with_dir_check(fig, plot_dir, filename)

def plot_radar_heatmap_metrics(df, plot_dir, level):

    # Set up radar chart
    fig = go.Figure()

    for agent in df.index:        
        fig.add_trace(go.Scatterpolar(
            r=df.loc[agent].values,
            theta=df.columns,
            fill='toself',
            name=agent
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Agent Behavior Comparison (Radar Plot)"
    )

    filename = f"{level}_heatmap_metrics_radar_plot"
    plotly_save_with_dir_check(fig, plot_dir, filename)
    # fig.show()

def plot_bar_heatmap_metrics(df, plot_dir, level):

    df_melted = df.melt(id_vars=["agent"], var_name="metric", value_name="value")

    fig = px.bar(
        df_melted,
        x="agent",
        y="value",
        color="metric",
        barmode="group",
        title="Agent Metrics Comparison"
    )
    fig.update_layout(xaxis_tickangle=-45)
   
    filename = f"{level}_heatmap_metrics_bar_plot"
    plotly_save_with_dir_check(fig, plot_dir, filename)

def plot_interactive_heatmap_metrics(df, plot_dir, level):
    # Create one trace per metric
    metrics = df.columns[1:]  # skip 'agent'
    fig = go.Figure()

    for i, metric in enumerate(metrics):
        visible = [i == j for j in range(len(metrics))]
        fig.add_trace(go.Bar(
            x=df["agent"],
            y=df[metric],
            name=metric,
            visible=visible[i]  # only show the first metric initially
        ))

    # Add dropdown buttons to toggle each metric
    dropdown_buttons = [
        dict(label=metric,
            method="update",
            args=[{"visible": [i == j for j in range(len(metrics))]},
                {"title": f"{metric} by Agent"}])
        for i, metric in enumerate(metrics)
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=1.05,
                xanchor="left",
                y=1,
                yanchor="top"
            )
        ],
        title=f"{metrics[0]} by Agent",
        xaxis=dict(title="Agent", tickangle=-45),
        yaxis=dict(title="Value"),
        showlegend=False
    )

    filename = f"{level}_heatmap_interactive_metrics_bar_plot"
    plotly_save_with_dir_check(fig, plot_dir, filename)

def plot_top_k_curves(curve_dictionary, plot_dir, level, plot_zoomed=False):
    fig = go.Figure()

    for agent in curve_dictionary:
        k_vals, concentrations = curve_dictionary[agent]
        if plot_zoomed:
            k_vals = k_vals[:int(len(k_vals)/5)] # 20%
        fig.add_trace(go.Scatter(
            x=k_vals * 100,  # Convert to percent
            y=concentrations,
            mode='lines',
            name=agent
        ))

    fig.update_layout(
        title="Top-K Concentration Curve per Agent",
        xaxis_title="Top-K % of Most Visited Locations",
        yaxis_title="Cumulative Movement Concentration",
        xaxis=dict(tickmode="linear", dtick=10),
        yaxis=dict(range=[0, 1]),
        width=800,
        height=500
    )


    filename = f"{level}_top_k_concentration_curves"
    filename = filename + "_reduced_range" if plot_zoomed else filename
    plotly_save_with_dir_check(fig, plot_dir, filename)

    if not plot_zoomed:
        plot_top_k_curves(curve_dictionary, plot_dir, level, plot_zoomed=True)
    

def new_column_names(current_names):
    column_names = {}

    for name in current_names:
        if 'bc' in name:
            split_name = name[(len('PPO_bc_')):].split('_20M')
            if 'only' in name:
                column_names[name] = name[:3] + "_BC_only_" + split_name[0]
            else:
                column_names[name] = name[:3] + "_20M_supervised_" + split_name[0]
        elif 'demo' in name:
            split_name = name.split(' ')
            column_names[name] = "Demo_Data_" + split_name[0]
        elif 'best' in name:
            split_name = name[(len('best_')):].split('_20M')
            column_names[name] = name[:4] + "_" + split_name[0]
        elif "20M" not in name:
            # PPO_15M_unsupervised_4
            # PPO_unsupervised_20M_agent_4
            # split_name = name.split("_")
            # if split_name[-1][-1].isdigit():
            # column_names[name] = name[:-(len(split_name[-1]) + 1)]
            column_names[name] = name
        else:
            split_name = name[(len('PPO_')):].split('_20M')
            column_names[name] = name[:len('PPO_')] + "20M_" + split_name[0]

    return column_names

def rename_columns(df):
    
    column_names = new_column_names(df.columns.tolist())

    df = df.rename(columns=column_names)

    return df

def main():

    demo_bk2_dir = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario/user_bk2s"
    demo_dir = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/demo_pickle_files/"
    agent_eval_dir = f"/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/training_logs/experiments/{EXP_ID}/saved_models/level_change_random"
    agent_dir = f"/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/{EXP_ID}/"
    plot_dir = f"{agent_dir}plots/"

    dataset_dir = "bc_datasets/"

    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir, exist_ok=True)
    
    if not os.path.exists(agent_dir):
        os.makedirs(agent_dir, exist_ok=True)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    statistics = {}

    results = {}
    all_levels = ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]
    data_generated = False

    table_types = ["all", "demo", "bc_only", "rl", "not_demo"]

    table_data = {}

    for table_type in table_types:
        table_data[table_type] = ""

    for level in all_levels:
        # get info from human demo data:
        results[level] = {}

        amalgam_demo_filename = f"{demo_dir}{level}_amalgam_demo_dataframe.pkl"

        expert_demo_filename = f"{demo_dir}{level}_expert_demo_dataframe.pkl"
        nonexpert_demo_filename = f"{demo_dir}{level}_nonexpert_demo_dataframe.pkl"

        # set up a FORCE_RELOAD bool 
        if not FORCE_RELOAD and not GENERATE_DATASETS and (os.path.isfile(amalgam_demo_filename) and os.path.isfile(expert_demo_filename) and os.path.isfile(nonexpert_demo_filename)):
            df = pd.read_pickle(amalgam_demo_filename)
            results[level]["amalgam demo data"] = df.to_dict()

            df = pd.read_pickle(expert_demo_filename)
            results[level]["expert demo data"] = df.to_dict()

            df = pd.read_pickle(nonexpert_demo_filename)
            results[level]["nonexpert demo data"] = df.to_dict()
        else:
            data_generated = True
            all_trajectories, all_states, all_actions, all_next_states, all_individual_dist_rewards, all_individual_score_rewards, all_dist_rewards, all_score_rewards, all_combined_rewards, all_action_distributions, all_max_scores, all_death_types, all_death_logs = load_in_demo_data(demo_bk2_dir, level)
            results[level]["amalgam demo data"] = {"trajectories": copy.deepcopy(all_trajectories), 
                                            "actions": copy.deepcopy(all_actions),
                                            "dist_rewards": all_dist_rewards,
                                            "score_rewards": all_score_rewards,
                                            "combined_rewards": all_combined_rewards,
                                            "action_distributions": all_action_distributions,
                                            "max_score": all_max_scores,
                                            "fall_ends": all_death_types["fall"],
                                            "enemy_ends": all_death_types["enemy"],
                                            "timeout_ends": all_death_types["timeout"],
                                            "flagpole_ends": all_death_types["flagpole"],
                                            "death_logs": all_death_logs
                                            }
            # print(all_death_types)

            df = pd.DataFrame.from_dict(results[level]["amalgam demo data"])

            dataframe_saving(df, amalgam_demo_filename)

            if GENERATE_DATASETS:
                generate_dataset_from_state_actions(dataset_dir, "amalgam", level, all_states, all_actions)
                generate_offline_rl_dataset(dataset_dir, "amalgam", level, all_states, all_actions, all_next_states, all_individual_dist_rewards, all_individual_score_rewards)

            data_generated = True
            expert_indices = []
            expert_states = []
            nonexpert_states = []
            expert_next_states = []
            nonexpert_next_states = []
            expert_individual_dist_rewards = []
            nonexpert_individual_dist_rewards = []
            expert_individual_score_rewards = []
            nonexpert_individual_score_rewards = []

            for i, trajectory in enumerate(all_trajectories):
                if distance_expert(trajectory):
                    expert_indices.append(i)
                    expert_states.append(all_states[i])
                    expert_individual_dist_rewards.append(all_individual_dist_rewards[i])
                    expert_individual_score_rewards.append(all_individual_score_rewards[i])
                    expert_next_states.append(all_next_states[i])
                else:
                    nonexpert_states.append(all_states[i])
                    nonexpert_individual_dist_rewards.append(all_individual_dist_rewards[i])
                    nonexpert_individual_score_rewards.append(all_individual_score_rewards[i])
                    nonexpert_next_states.append(all_next_states[i])

            results[level]["expert demo data"] = {}
            results[level]["nonexpert demo data"] = {}

            for df_key in results[level]["amalgam demo data"]:
                if "_ends" in df_key:
                    continue
                all_data = results[level]["amalgam demo data"][df_key]
                expert_data, nonexpert_data = split_by_indices(all_data, expert_indices)
                results[level]["expert demo data"][df_key] = copy.deepcopy(expert_data)
                results[level]["nonexpert demo data"][df_key] = copy.deepcopy(nonexpert_data)
            
            exp_death_types = {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0}

            for death_log in results[level]["expert demo data"]["death_logs"]:
                if death_log:
                    exp_death_types[death_log["type"]] += 1

            for key in exp_death_types:
                results[level]["expert demo data"][f"{key}_ends"] = exp_death_types[key]

            nonexp_death_types = {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0}

            for death_log in results[level]["nonexpert demo data"]["death_logs"]:
                if death_log:
                    nonexp_death_types[death_log["type"]] += 1

            for key in nonexp_death_types:
                results[level]["nonexpert demo data"][f"{key}_ends"] = nonexp_death_types[key]

            df = pd.DataFrame.from_dict(results[level]["expert demo data"])
            dataframe_saving(df, expert_demo_filename)

            df = pd.DataFrame.from_dict(results[level]["nonexpert demo data"])
            dataframe_saving(df, nonexpert_demo_filename)

            if GENERATE_DATASETS:
                expert_actions = results[level]["expert demo data"]["actions"]
                nonexpert_actions = results[level]["nonexpert demo data"]["actions"]
                generate_dataset_from_state_actions(dataset_dir, "expert_distance", level, expert_states, expert_actions)
                generate_dataset_from_state_actions(dataset_dir, "nonexpert_distance", level, nonexpert_states, nonexpert_actions)
                
                generate_offline_rl_dataset(dataset_dir, "expert_distance", level, expert_states, expert_actions, expert_next_states, expert_individual_dist_rewards, expert_individual_score_rewards)
                generate_offline_rl_dataset(dataset_dir, "nonexpert_distance", level, nonexpert_states, nonexpert_actions, nonexpert_next_states, nonexpert_individual_dist_rewards, nonexpert_individual_score_rewards)
               
        agent_demo_filename = f"{agent_dir}{level}_agent_dataframe.pkl"

        if not FORCE_RELOAD and not FORCE_RELOAD_AGENTS and os.path.isfile(agent_demo_filename):
            df = pd.read_pickle(agent_demo_filename)

            results[level].update(df.to_dict())
        else:
            data_generated = True
            agents = load_in_agent_data(agent_eval_dir, level)

            print(agents.keys())

            results[level].update(agents)
            
            df = pd.DataFrame.from_dict(agents)
            dataframe_saving(df, agent_demo_filename)

        file_name = f"{agent_dir}evaluation_dataframe_{level}.pkl"
        
        if not FORCE_RELOAD and (not data_generated and os.path.isfile(file_name)):
            eval_df = pd.read_pickle(file_name)
        else:
            eval_df = pd.DataFrame.from_dict(results[level])
            eval_df = eval_df.loc[:, ~eval_df.applymap(lambda x: isinstance(x, list) and len(x) == 0).all()]
            eval_df.to_pickle(file_name)

        file_name = f'{agent_dir}{level}_stats_df.pkl'

        if not FORCE_RELOAD and (not data_generated and os.path.isfile(file_name)):
            df = pd.read_pickle(file_name)
            statistics[level] = df

        else:
            statistics[level] = compute_statistics(eval_df)

            df = pd.DataFrame.from_dict(statistics[level])
            # statistics[level] = df
            print(df)

            df = rename_columns(df)

            df = df.reindex(sorted(df.columns), axis=1)
        
            df.to_pickle(file_name)

            statistics[level] = df

        if PLOTS:
            mean_plots(df, level, True, plot_dir)
            # # mean_rewards_plot(df, level, False, plot_dir)
            plot_action_distributions(df, level, plot_dir)

        if MAKE_TABLE:
            indices_to_drop = []

            for index in df.index.tolist():
                if "mean" not in index:
                    indices_to_drop.append(index)
                    # df.drop(index=index)

            df = df.drop(index=indices_to_drop)

            for table_type in table_types:

                if table_type != "all":
                    columns_to_drop = []
                    for column in df.columns.tolist():
                        if table_type == "rl":
                            if "bc_only" in column.lower() or "demo" in column.lower():
                                columns_to_drop.append(column)
                        elif table_type == "not_demo":
                            if "demo" in column.lower():
                                columns_to_drop.append(column)
                        else:
                            if table_type not in column.lower():
                                columns_to_drop.append(column)
                        

                    new_df = df.drop(columns=columns_to_drop)
                else:
                    new_df = df

                new_df = new_df.astype(float)
                styled_df = new_df.style.background_gradient(axis=1, cmap='RdYlGn', low=0).format("{:.0f}")

                table_data[table_type] += f"{level}<br/>{styled_df.render()}<br/><br/>"

        if PLOT_HEATMAPS:
            column_names = {}
            heatmap_metrics = {}
            demo_heatmap_metrics = {}

            top_k_curve_data = {}

            column_names = new_column_names(list(results[level].keys()))

            demo_loaded = False
            # load in demo metrics
            file_name = f'{demo_dir}{level}_heatmap_metrics_df.pkl'
            if not FORCE_RELOAD and os.path.isfile(file_name):
                demo_heatmap_metrics = pd.read_pickle(file_name)
                demo_loaded = True

            for agent_type in results[level]:
                agent_trajectories = results[level][agent_type]["trajectories"]
                # print(agent_trajectories)
                if isinstance(agent_trajectories, dict):
                    agent_trajectories = list(agent_trajectories.values())
                # elif isinstance(agent_trajectories, list):  # If already a list, convert to NumPy array
                #     agent_trajectories = np.array(agent_trajectories)

                if "demo" in agent_type.lower():
                    if not demo_loaded:
                        _, metrics = plot_mario_heatmap(agent_trajectories, demo_dir, level, f"{column_names[agent_type]}")
                        demo_heatmap_metrics[agent_type] = metrics

                        # top_k_curve_data[agent_type] = metrics["top_k_values"]

                    # top_k_curve_data[agent_type] = demo_heatmap_metrics.loc[agent_type]["top_k_values"]
                elif "bc_only" in agent_type.lower():
                    continue
                else:
                    _, metrics = plot_mario_heatmap(agent_trajectories, plot_dir, level, f"{column_names[agent_type]}")
                    heatmap_metrics[agent_type] = metrics
                    top_k_curve_data[agent_type] = metrics["top_k_values"]
            
            plot_top_k_curves(top_k_curve_data, agent_dir, level)

            if type(demo_heatmap_metrics) is dict:
                demo_heatmap_metrics = pd.DataFrame.from_dict(demo_heatmap_metrics, orient='index')

            heatmap_df = pd.DataFrame.from_dict(heatmap_metrics, orient='index')
            heatmap_df = pd.concat([heatmap_df, demo_heatmap_metrics])

            file_name = f'{agent_dir}{level}_heatmap_metrics_df.pkl'
            dataframe_saving(heatmap_df, file_name)

            if not demo_loaded:
                file_name = f'{demo_dir}{level}_heatmap_metrics_df.pkl'
                dataframe_saving(demo_heatmap_metrics, file_name)
            
            statistics[level] = pd.concat([statistics[level], heatmap_df])
            # plot_interactive_heatmap_metrics(heatmap_df, plot_dir, level)
            # plot_bar_heatmap_metrics(heatmap_df, plot_dir, level)
            # plot_radar_heatmap_metrics(heatmap_df, plot_dir, level)


    if PLOTS:
        agent_types = []

        for agent_type in statistics[level].columns.tolist():
            if "demo" in agent_type.lower() or "bc_only" in agent_type.lower():
                continue
            else:
                agent_types.append(agent_type)

        plot_level_scatters(statistics, plot_dir, agent_types)

    if MAKE_TABLE:
        for table_type in table_data:
            with open(f"{plot_dir}mean_reward_table_{table_type}.html", "w", encoding="utf-8") as f:
                f.write(table_data[table_type])


for exp_id in ["25_tuned_exp_params"]:#, "100_tuned_exp_params", "1000_tuned_exp_params", "100_score_tuned_params", "100_combined_tuned_params", "25_score_tuned_exp_params", "25_combined_tuned_params"]:
# for exp_id in ["100_tuned_exp_params", "1000_tuned_exp_params", "100_score_tuned_params", "100_combined_tuned_params", "25_score_tuned_exp_params", "25_combined_tuned_params"]:

    EXP_ID = exp_id

    if "score" in exp_id:
        TRAINING_REWARD = "Score"
    elif "combined" in exp_id:
        TRAINING_REWARD = "Combined"
    else:
        TRAINING_REWARD = "Distance"

    main()

# EXP_ID = "25_tuned_exp_params"
# TRAINING_REWARD = "Distance"
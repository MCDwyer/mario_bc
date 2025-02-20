import retro
import numpy as np
import copy
import os
import pandas as pd

MAX_X = 3710
MAX_Y = 768

DISTANCE_THRESHOLD = MAX_X/2

def extract_info_from_bk2s(bk2_file):
    try:
        movie = retro.Movie(bk2_file)
    except:
        print(bk2_file)
        movie = retro.Movie(bk2_file)

    movie.step()

    env = retro.make(game=movie.get_game())#, obs_type=retro.Observations.RAM)
    # env = ProcessedFrame(env)
    env.initial_state = movie.get_state()
    
    env.reset()
    
    trajectories = []
    action_distribution = np.zeros(13)
    actions = []
    reward = 0

    step = 0
    level = None

    prev_position = 40

    while movie.step():
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))
        obs, _, done, info = env.step(keys)
        step += 1

        info = copy.deepcopy(info)

        if level is None:
            level = info["level"]

        x = info["x_frame"]*256 + info["x_position_in_frame"]

        y = ((info["y_frame"]*256) + info["y_position_in_frame"])

        if done or info["player_state"] == 11 or info["level"] != level or y > MAX_Y:# < -432:# or info["viewport_position"] > 1: #y < -432:# or info["player_dead"] != 32:# or y < -432:

            break

        reward += horizontal_reward(x, prev_position)
        prev_position = x

        if step%4 == 0: # only want every 4 timesteps?
            while info["player_state"] != 8:# and info["player_state"] != 11:
                # keep going as non playable bit?
                obs, _, done, info = env.step(keys)

            if x != 0 and y != 0:
                # y = 1024 - y
                action = map_from_retro_action(keys)
                trajectories.append([x, y])
                actions.append(action)
                action_distribution[int(action)] += 1

    env.close()

    # check that there is actually a trajectory here:
    if trajectories:
        trajectories = np.array(trajectories)

        x_coords = trajectories[:, 0]
        y_coords = trajectories[:, 1]

        if len(set(x_coords)) == 1 or len(set(actions)) == 1:
            return None, None, None, None

        return trajectories, actions, reward, action_distribution
    
    return None, None, None, None

def horizontal_reward(current_horizontal_position, prev_position=40):

    reward = current_horizontal_position - prev_position
        
    return reward


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
    all_rewards = []
    all_action_distributions = []
    # combined_action_distribution = np.zeros(13)

    for filename in full_file_list:
        if filename.endswith(".bk2"):
            full_filename = f"{demo_dir_level}/{filename}"
            trajectories, actions, reward, action_distribution = extract_info_from_bk2s(full_filename)

            if trajectories is not None:
                all_trajectories.append(trajectories)
                all_actions.append(actions)
                all_rewards.append(reward)
                all_action_distributions.append(action_distribution)
                # combined_action_distribution += action_distribution

    return all_trajectories, all_actions, all_rewards, all_action_distributions#, combined_action_distribution

def load_in_agent_data(agent_dir, level, check_for=""):
    full_file_list = filter(
        lambda f: os.path.isdir(os.path.join(agent_dir, f)),  # Use full path
        os.listdir(agent_dir)  # List only filenames
    )

    # Convert filter object to a list
    full_file_list = list(full_file_list)

    agents = {}
    print(retro.data.get_romfile_path('SuperMarioBros-Nes'))  # Example game
    for directory in full_file_list:
        # agent type
        agent_type = directory[-2] # remove index and _

        if check_for in agent_type:

            if agent_type not in agents:
                agents[agent_type] = {"trajectories": [],
                                    "actions": [],
                                    "rewards": [],
                                    "action_distributions": []}

            full_agent_dir = f"{agent_dir}/{directory}/evaluation/{level}/bk2_files"

            bk2s_file_list = os.listdir(full_agent_dir)

            all_trajectories = []
            all_actions = []
            all_rewards = []
            all_action_distributions = []

            for bk2_dir in bk2s_file_list:
                bk2_filename = os.listdir(f"{full_agent_dir}/{bk2_dir}")[0]
                bk2_filepath = f"{full_agent_dir}/{bk2_filename}"

                trajectories, actions, reward, action_distribution = extract_info_from_bk2s(bk2_filepath)

                if trajectories is not None:
                    all_trajectories.append(trajectories)
                    all_actions.append(actions)
                    all_rewards.append(reward)
                    all_action_distributions.append(action_distribution)

            agents[agent_type]["trajectories"] = agents[agent_type]["trajectories"] + copy.deepcopy(all_trajectories)
            agents[agent_type]["actions"] = agents[agent_type]["actions"] + copy.deepcopy(all_actions)
            agents[agent_type]["rewards"] = agents[agent_type]["rewards"] + copy.deepcopy(all_rewards)
            agents[agent_type]["action_distributions"] = agents[agent_type]["action_distributions"] + copy.deepcopy(all_action_distributions)

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

def main():

    demo_dir = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario/user_bk2s"
    agent_dir = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/training_logs/experiments/tuned_exp_params/saved_models/level_change_random"
    exp_agent_dir = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/training_logs/experiments/tuned_params/saved_models/level_change_random"

    results = {}
    all_levels = ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]

    for level in all_levels:
        # get info from human demo data:
        results[level] = {}

        amalgam_demo_filename = f"{level}_amalgam_demo_dataframe.pkl"

        if os.path.isfile(amalgam_demo_filename):
            df = pd.read_pickle(amalgam_demo_filename)
            results[level]["amalgam demo data"] = df.to_dict()
        else:
            all_trajectories, all_actions, all_rewards, all_action_distributions = load_in_demo_data(demo_dir, level)
            results[level]["amalgam demo data"] = {"trajectories": copy.deepcopy(all_trajectories), 
                                            "actions": copy.deepcopy(all_actions), 
                                            "rewards": copy.deepcopy(all_rewards), 
                                            "action_distributions": copy.deepcopy(all_action_distributions)}#, 
                                            # "mean_reward": mean_reward,
                                            # "combined_action_distributions": copy.deepcopy(combined_action_distribution)}

            df = pd.DataFrame.from_dict(results[level]["amalgam demo data"])

            dataframe_saving(df, amalgam_demo_filename)

        expert_demo_filename = f"{level}_expert_demo_dataframe.pkl"
        nonexpert_demo_filename = f"{level}_nonexpert_demo_dataframe.pkl"

        if os.path.isfile(expert_demo_filename) and os.path.isfile(nonexpert_demo_filename):
            df = pd.read_pickle(expert_demo_filename)
            results[level]["expert demo data"] = df.to_dict()

            df = pd.read_pickle(nonexpert_demo_filename)
            results[level]["nonexpert demo data"] = df.to_dict()

        else:
            expert_indices = []

            for i, trajectory in enumerate(all_trajectories):
                if distance_expert(trajectory):
                    expert_indices.append(i)

            results[level]["expert demo data"] = {}
            results[level]["nonexpert demo data"] = {}

            for key in results[level]["amalgam demo data"]:
                all_data = results[level]["amalgam demo data"][key]
                expert_data, nonexpert_data = split_by_indices(all_data, expert_indices)
                results[level]["expert demo data"][key] = copy.deepcopy(expert_data)
                results[level]["nonexpert demo data"][key] = copy.deepcopy(nonexpert_data)
            
            df = pd.DataFrame.from_dict(results[level]["expert demo data"])
            dataframe_saving(df, expert_demo_filename)

            df = pd.DataFrame.from_dict(results[level]["nonexpert demo data"])
            dataframe_saving(df, nonexpert_demo_filename)

        agent_demo_filename = f"{level}_agent_demo_dataframe.pkl"

        if os.path.isfile(agent_demo_filename):
            df = pd.read_pickle(agent_demo_filename)
            results[level].update(df.to_dict())
        else:

            agents = load_in_agent_data(agent_dir, level, check_for="_expert")
            results[level].update(agents)
            
            df = pd.DataFrame.from_dict(agents)
            dataframe_saving(df, agent_demo_filename)

        agent_demo_filename = f"{level}_exp_agent_demo_dataframe.pkl"

        if os.path.isfile(agent_demo_filename):
            df = pd.read_pickle(agent_demo_filename)
            results[level].update(df.to_dict())
        else:

            agents = load_in_agent_data(exp_agent_dir, level)
            results[level].update(agents)
            
            df = pd.DataFrame.from_dict(agents)
            dataframe_saving(df, agent_demo_filename)

        eval_df = pd.DataFrame.from_dict(results[level])
        file_name = f"evaluation_dataframe_{level}.pkl"
        eval_df.to_pickle(file_name) 

        print(eval_df)


# def score_expert(info, trajectory):

#     score = info[-1]["score"]

#     return score > SCORE_THRESHOLD

# def speed_expert(info, trajectory):
#     time = 400 - np.min(info["time"])

#     x = trajectory[:, 0]

#     total_distance = 0
#     for k in range(1, len(x)):
#         total_distance += abs(x[k] - x[k - 1])

#     avg_speed = total_distance/time

#     return avg_speed > SPEED_THRESHOLD

# def get_all_metrics(useful_info, is_expert_function):
#     metrics = {}
#     nonexpert_metrics = {}
#     expert_metrics = {}

#     expert_trajectories = {}
#     nonexpert_trajectories = {}

#     expert_obs = {}
#     nonexpert_obs = {}

#     expert_info = {}
#     nonexpert_info = {}

#     for level in useful_info:
#         metrics[level] = {"high_score": [], "furthest_distance": [], "highest_distance": [], "average_height": [], "average_speed": [], "time_taken": []}

#         expert_metrics[level] = copy.deepcopy(metrics[level])
#         expert_trajectories[level] = []
#         expert_obs[level] = []
#         expert_info[level] = []

#         nonexpert_metrics[level] = copy.deepcopy(metrics[level])
#         nonexpert_trajectories[level] = []
#         nonexpert_obs[level] = []
#         nonexpert_info[level] = []

#         useful_infos = []

#         for i in range(len(useful_info[level])):

#             combined_dict = {key: [] for key in useful_info[level][i][0].keys()}
#             # Iterate over each dictionary in the list
#             for d in useful_info[level][i]:
#                 for key, value in d.items():
#                     combined_dict[key].append(value)
        
#             useful_infos.append(combined_dict) 

#         for j, infos in enumerate(useful_infos):

#             metrics[level] = get_trajectory_metrics(metrics[level], useful_trajectories[level][j], infos)

#             if is_expert_function(infos, useful_trajectories[level][j]):
#                 expert_metrics[level] = get_trajectory_metrics(expert_metrics[level], useful_trajectories[level][j], infos)
#                 expert_info[level].append(useful_info[level][j])
#                 expert_trajectories[level].append(useful_trajectories[level][j])
#                 expert_obs[level].append(useful_obs[level][j])
#             else:
#                 nonexpert_metrics[level] = get_trajectory_metrics(nonexpert_metrics[level], useful_trajectories[level][j], infos)
#                 nonexpert_info[level].append(useful_info[level][j])
#                 nonexpert_trajectories[level].append(useful_trajectories[level][j])
#                 nonexpert_obs[level].append(useful_obs[level][j])

#     return metrics, expert_trajectories, nonexpert_trajectories, expert_obs, nonexpert_obs, expert_metrics, nonexpert_metrics, expert_info, nonexpert_info

main()
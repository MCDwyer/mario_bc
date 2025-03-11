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

MAX_X = 3710
MAX_Y = 768
MARIO_X = 13
MARIO_Y = 16

DISTANCE_THRESHOLD = MAX_X/2

COLOUR_SCHEME = px.colors.qualitative.Plotly

EXP_ID = "tuned_exp_params"
TRAINING_REWARD = "Distance"

# EXP_ID = "score_tuned_params"
# TRAINING_REWARD = "Score"

# EXP_ID = "combined_tuned_params"
# TRAINING_REWARD = "Combined"

def process_observation(obs):
    # Convert the frame to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # Resize the frame to 84x84
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # Add a channel dimension
    obs = np.expand_dims(obs, axis=-1)

    # obs = np.transpose(obs, (0, 3, 1, 2))
    return obs

def generate_dataset_from_state_actions(dir, filename, level, all_states, all_actions):

    # combine the states and actions together?

    stacked_states = np.array(all_states[0])

    # stacked_states = stacked_states.reshape(stacked_states.shape[3], stacked_states.shape[0], stacked_states.shape[1], stacked_states.shape[2])

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

    while movie.step():
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))

        obs, _, done, info = env.step(keys)
        step += 1

        info = copy.deepcopy(info)

        if level is None:
            level = info["level"]

        if step%4 == 0: # only want every 4 timesteps?
            state_change = False

            last_info = copy.deepcopy(info)
            death_log = {}

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

            # if info["level"] != level:
            #     level = info["level"]
            #     print(env.statename, info["level"])

            if info["lives"] != 2 or info["level"] != level:
                death_type = death_log["type"]
                break
            # if done or info["player_state"] == 11 or info["level"] != level or y > MAX_Y:# < -432:# or info["viewport_position"] > 1: #y < -432:# or info["player_dead"] != 32:# or y < -432:
            #     print(env.statename, info["level"])
            #     print(info["lives"])
            #     break

            # if done:
            #     break

            if x != 0 and y != 0:
                # y = 1024 - y
                action = map_from_retro_action(keys)
                trajectories.append([x, y])
                processed_obs = process_observation(prev_state)
                states.append(processed_obs)
                actions.append(action)
                action_distribution[int(action)] += 1

            if state_change:
                dist_reward = 0
                score_reward = 0
                state_change = False
            else:
                dist_reward = (x - prev_position)
                score_reward = ((info['score']*10) - prev_score)
                # score_reward = ((info['score']) - prev_score)
                

            total_dist_reward += dist_reward
            total_score_reward += score_reward
            total_combined_reward += (dist_reward/2 + score_reward/2)

            prev_position = x
            prev_score = info['score']*10
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
            return None, None, None, None, None, None, None, None, None, None

        return trajectories, states, actions, total_dist_reward, total_score_reward, total_combined_reward, action_distribution, max_score, death_type, death_log
    
    return None, None, None, None, None, None, None, None, None, None

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
            trajectories, states, actions, total_dist_reward, total_score_reward, total_combined_reward, action_distribution, max_score, death_type, death_log = extract_info_from_bk2s(full_filename, level)

            if trajectories is not None:
                all_trajectories.append(trajectories)
                all_actions.append(actions)
                all_states.append(states)
                all_dist_rewards.append(total_dist_reward)
                all_score_rewards.append(total_score_reward)
                all_combined_rewards.append(total_combined_reward)
                all_action_distributions.append(action_distribution)
                all_max_scores.append(max_score)
                all_death_types[death_type] += 1
                all_death_logs.append(death_log)
                # combined_action_distribution += action_distribution

    return all_trajectories, all_states, all_actions, all_dist_rewards, all_score_rewards, all_combined_rewards, all_action_distributions, all_max_scores, all_death_types, all_death_logs#, combined_action_distribution

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
                                "death_types": {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0},
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
                        if df_key == "death_types":
                            agents[agent_type][df_key] = {key: df_dict[df_key][key] + agents[agent_type][df_key][key] for key in agents[agent_type][df_key]}
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

        dist_rewards = None
        score_rewards = None
        
        for reward_type in ["dist_rewards", "score_rewards", "combined_rewards"]:
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
            
        # Extract action distributions dictionary
        action_dist_data = df.loc["action_distributions", key]
        if isinstance(action_dist_data, dict):
            combined_action_dist = np.sum(np.array(list(action_dist_data.values())), axis=0)  # Sum across episodes
        elif isinstance(rewards_data, list):  # If already a list, convert to NumPy array
            combined_action_dist = np.sum(np.array(action_dist_data), axis=0)
        else:
            raise TypeError(f"Unexpected type {type(action_dist_data)} for action distributions in {key}")


        results[key]["combined_action_distribution"] = combined_action_dist

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

        if "nonexpert" in agent_type.lower():
            colours.append(COLOUR_SCHEME[0])
        elif "expert" in agent_type.lower():
            colours.append(COLOUR_SCHEME[1])
        elif "amalgam" in agent_type.lower():
            colours.append(COLOUR_SCHEME[2])
        else:
            colours.append(COLOUR_SCHEME[3])

    return colours, pattern_shape_sequence, marker_shapes


def mean_rewards_plot(df, level, with_errors=True, plot_dir="", training_reward=TRAINING_REWARD):

    sig_figs = 3
    all_rewards_types = []

    for row_name in df.index.tolist():
        if "mean" in row_name:
            clean_row_name = row_name[len("mean_"):]
            all_rewards_types.append(clean_row_name)

    subplot_fig = make_subplots(3, 1, subplot_titles=all_rewards_types, shared_xaxes=True, specs = [[{}], [{}], [{}]],
                          vertical_spacing = 0.05)

    colours = []
    pattern_shape_sequence = []

    colours, pattern_shape_sequence, _ = plot_patterns(df.columns.tolist())

    for i, reward_type in enumerate(all_rewards_types):
        # Extract mean and std deviation values
        mean_rewards = df.loc[f"mean_{reward_type}"]
        std_rewards = df.loc[f"std_{reward_type}"]

        # Create a DataFrame for Plotly
        plot_df = pd.DataFrame({
            "Agent Type": mean_rewards.index,  # Column names as agent types
            "Mean Reward": mean_rewards.values,
            "Std Reward": std_rewards.values
        })

        if with_errors:
            # Create bar plot with error bars
            fig = px.bar(
                plot_df,
                x="Agent Type",
                y="Mean Reward",
                error_y="Std Reward",
                title=f"Mean Reward ({reward_type}) with Standard Deviation per Agent for Level: {level}",
                labels={"Mean Reward": "Mean Reward", "Agent Type": "Agent Type"},
                color="Agent Type",
                color_discrete_sequence = colours,
                pattern_shape="Agent Type",
                pattern_shape_sequence=pattern_shape_sequence,
                text_auto=f'.{sig_figs}s'
            )

            subplot_fig.add_trace(
                go.Bar(
                    x=plot_df["Agent Type"],
                    y=plot_df["Mean Reward"],
                    error_y=dict(type='data', array=plot_df["Std Reward"], visible=True),
                    # name=reward_type,
                    marker=dict(
                        color=colours,
                        pattern_shape=pattern_shape_sequence,  # Different patterns per bar
                        pattern_fgcolor="grey"
                    ),
                    text=[f"{val:.{sig_figs}g}" for val in plot_df["Mean Reward"]],
                    textposition='outside',
                    showlegend=False),
                    row=(i+1),
                    col=1)

            plot_path = f"{level}_mean_{reward_type}_with_errors"
        else:
            # Create bar plot with error bars
            fig = px.bar(
                plot_df,
                x="Agent Type",
                y="Mean Reward",
                title=f"Mean Reward ({reward_type}) per Agent for Level: {level}",
                labels={"Mean Reward": "Mean Reward", "Agent Type": "Agent Type"},
                color="Agent Type",
                color_discrete_sequence = colours,
                pattern_shape="Agent Type",
                pattern_shape_sequence=pattern_shape_sequence,
                text_auto=f'.{sig_figs}s'
            )

            subplot_fig.add_trace(
                go.Bar(
                    x=plot_df["Agent Type"],
                    y=plot_df["Mean Reward"],
                    # name=reward_type,
                    marker=dict(
                        color=px.colors.qualitative.Pastel[:len(df.columns.tolist())],
                        pattern_shape=pattern_shape_sequence,  # Different patterns per bar
                        pattern_fgcolor="grey"
                    ),
                    text=[f"{val:.{sig_figs}g}" for val in plot_df["Mean Reward"]],
                    textposition="outside",
                    showlegend=False),
                    row=(i+1),
                    col=1)

            plot_path = f"{level}_mean_{reward_type}"

        fig.update_layout(showlegend=False)#, textposition="outside", cliponaxis=False)
        plotly_save_with_dir_check(fig, plot_dir, plot_path)

    plot_path = f"{level}_mean_rewards"
    plot_path += "_with_errors" if with_errors else ""
    plot_path += "_subplot"

    subplot_fig.update_layout(
        margin=dict(t=50, b=40, l=30, r=10),  # Reduce top margin (t), adjust others as needed
        height=800,  # Increase figure height
        width=800,   # Increase figure width
        title_text=f"Mean Rewards for Level: {level} - Trained with {training_reward} reward",
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

def get_mario_heatmap(trajectories, mario=True):

    # from level image
    x_grid_max = MAX_X + MARIO_X #4000
    y_grid_max = MAX_Y + MARIO_Y #800

    heatmap = np.zeros(((y_grid_max + 1), (x_grid_max + 1)))
    print(heatmap.shape)

    for trajectory in trajectories:

        trajectory_heatmap = np.zeros_like(heatmap)
        
        for state in trajectory:
            x, y = state
        
            if not mario:
                if x < MAX_X and y < MAX_Y:
                    trajectory_heatmap[int(y), int(x)] += 1
            else:
                for i in range(MARIO_X):
                    new_x = x + i
                    for j in range(MARIO_Y):
                        new_y = y + j
                
                        if new_x < MAX_X and new_y < MAX_Y:
                            trajectory_heatmap[int(new_y), int(new_x)] += 1
            
        trajectory_heatmap = np.clip(trajectory_heatmap, 0, 1)
        heatmap = heatmap + trajectory_heatmap

    heatmap = heatmap[::-1, :]

    return heatmap, x_grid_max, y_grid_max

def colorbar(n):
    return dict(
        tick0 = 0,
        title = "Log colour scale",
        tickmode = "array",
        tickvals = np.linspace(0, n, n+1),
        ticktext = 10**np.linspace(0, n, n+1))

def plot_mario_heatmap(trajectories, plot_path, level, info="", mario=True, max_value=None, colour_scale="Hot", x_axis_max=MAX_X, y_axis_max=MAX_Y):
    
    heatmap, x_grid_max, y_grid_max = get_mario_heatmap(trajectories, mario)
    
    n = int(np.round(np.log10(np.max(heatmap))))

    log_heatmap = np.log10(heatmap)
    log_heatmap = np.nan_to_num(log_heatmap)
    
    heatmap = heatmap/np.sum(heatmap)

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

    return max_value


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

FORCE_RELOAD = True
GENERATE_DATASETS = False
PLOTS = False
PLOT_HEATMAPS = False
MAKE_TABLE = False

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
        if not FORCE_RELOAD and (os.path.isfile(amalgam_demo_filename) and os.path.isfile(expert_demo_filename) and os.path.isfile(nonexpert_demo_filename)):
            df = pd.read_pickle(amalgam_demo_filename)
            results[level]["amalgam demo data"] = df.to_dict()

            df = pd.read_pickle(expert_demo_filename)
            results[level]["expert demo data"] = df.to_dict()

            df = pd.read_pickle(nonexpert_demo_filename)
            results[level]["nonexpert demo data"] = df.to_dict()
        else:
            data_generated = True
            all_trajectories, all_states, all_actions, all_dist_rewards, all_score_rewards, all_combined_rewards, all_action_distributions, all_max_scores, all_death_types, all_death_logs = load_in_demo_data(demo_bk2_dir, level)
            results[level]["amalgam demo data"] = {"trajectories": copy.deepcopy(all_trajectories), 
                                            "actions": copy.deepcopy(all_actions),
                                            "dist_rewards": all_dist_rewards,
                                            "score_rewards": all_score_rewards,
                                            "combined_rewards": all_combined_rewards,
                                            "action_distributions": all_action_distributions,
                                            "max_score": all_max_scores,
                                            "death_types": all_death_types,
                                            "death_logs": all_death_logs
                                            }
            print(all_death_types)

            df = pd.DataFrame.from_dict(results[level]["amalgam demo data"])

            dataframe_saving(df, amalgam_demo_filename)

            if GENERATE_DATASETS:
                generate_dataset_from_state_actions(dataset_dir, "amalgam", level, all_states, all_actions)

            data_generated = True
            expert_indices = []
            expert_states = []
            nonexpert_states = []

            for i, trajectory in enumerate(all_trajectories):
                if distance_expert(trajectory):
                    expert_indices.append(i)
                    expert_states.append(all_states[i])
                else:
                    nonexpert_states.append(all_states[i])

            results[level]["expert demo data"] = {}
            results[level]["nonexpert demo data"] = {}

            for df_key in results[level]["amalgam demo data"]:
                if df_key == "death_types":
                    continue
                all_data = results[level]["amalgam demo data"][df_key]
                expert_data, nonexpert_data = split_by_indices(all_data, expert_indices)
                results[level]["expert demo data"][df_key] = copy.deepcopy(expert_data)
                results[level]["nonexpert demo data"][df_key] = copy.deepcopy(nonexpert_data)
            
            exp_death_types = {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0}

            for death_log in results[level]["expert demo data"]["death_logs"]:
                exp_death_types[death_log["type"]] += 1

            results[level]["expert demo data"]["death_types"] = exp_death_types
            
            nonexp_death_types = {"fall": 0, "enemy": 0, "flagpole": 0, "timeout": 0}

            for death_log in results[level]["nonexpert demo data"]["death_logs"]:
                nonexp_death_types[death_log["type"]] += 1

            results[level]["nonexpert demo data"]["death_types"] = nonexp_death_types

            print(exp_death_types)
            print(nonexp_death_types)

            df = pd.DataFrame.from_dict(results[level]["expert demo data"])
            dataframe_saving(df, expert_demo_filename)

            df = pd.DataFrame.from_dict(results[level]["nonexpert demo data"])
            dataframe_saving(df, nonexpert_demo_filename)

            if GENERATE_DATASETS:
                generate_dataset_from_state_actions(dataset_dir, "expert_distance", level, expert_states, results[level]["expert demo data"]["actions"])
                generate_dataset_from_state_actions(dataset_dir, "nonexpert_distance", level, nonexpert_states, results[level]["nonexpert demo data"]["actions"])

        agent_demo_filename = f"{agent_dir}{level}_agent_dataframe.pkl"

        if not FORCE_RELOAD and os.path.isfile(agent_demo_filename):
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
            print(df.head())

            column_names = {}

            for name in df.columns.tolist():
                if 'bc' in name:
                    split_name = name[(len('PPO_bc_')):].split('_20M')
                    if 'only' in name:
                        column_names[name] = name[:3] + "_BC_only_" + split_name[0]
                    else:
                        column_names[name] = name[:3] + "_" + split_name[0]
                elif 'demo' in name:
                    split_name = name.split(' ')
                    column_names[name] = "Demo_Data_" + split_name[0]
                else:
                    split_name = name[(len('PPO_')):].split('_20M')
                    column_names[name] = name[:3] + "_" + split_name[0]

            df = df.rename(columns=column_names)

            df = df.reindex(sorted(df.columns), axis=1)
        
            df.to_pickle(file_name)

            statistics[level] = df

        if PLOTS:
            mean_rewards_plot(df, level, True, plot_dir)
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
            
            for name in results[level]:
                if 'bc' in name:
                    split_name = name[(len('PPO_bc_')):].split('_20M')
                    if 'only' in name:
                        column_names[name] = name[:3] + "_BC_only_" + split_name[0]
                    else:
                        column_names[name] = name[:3] + "_" + split_name[0]
                elif 'demo' in name:
                    split_name = name.split(' ')
                    column_names[name] = "Demo_Data_" + split_name[0]
                else:
                    split_name = name[(len('PPO_')):].split('_20M')
                    column_names[name] = name[:3] + "_" + split_name[0]

            for agent_type in results[level]:
                agent_trajectories = results[level][agent_type]["trajectories"]
                # print(agent_trajectories)
                if isinstance(agent_trajectories, dict):
                    agent_trajectories = list(agent_trajectories.values())
                # elif isinstance(agent_trajectories, list):  # If already a list, convert to NumPy array
                #     agent_trajectories = np.array(agent_trajectories)

                if "demo" in agent_type.lower() or "bc_only" in agent_type.lower():
                    continue
                    plot_mario_heatmap(agent_trajectories, demo_dir, level, f"{column_names[agent_type]}")
                else:
                    plot_mario_heatmap(agent_trajectories, plot_dir, level, f"{column_names[agent_type]}")


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

main()
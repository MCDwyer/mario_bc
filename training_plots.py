import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess

COLOUR_SCHEME = px.colors.qualitative.Set1# + px.colors.qualitative.Pastel

ACTION_MAPPING = {
    0: "UP", 
    1: "DOWN", 
    2: "LEFT", 
    3: "RIGHT", 
    4: "JUMP", 
    5: "SHIFT", 
    6: "LEFT + JUMP", 
    7: "RIGHT + JUMP", 
    8: "LEFT + SHIFT",
    9: "RIGHT + SHIFT",
    10: "LEFT + SHIFT + JUMP",
    11: "RIGHT + SHIFT + JUMP", 
    12: "NO ACTION"
}


# Specify the tags you want to extract
TAGS = {
    "Evaluation Episode Reward": 'eval/mean_reward',
    "Evaluation Episode Length": 'eval/mean_ep_length',
    "Episode Reward": "rollout/ep_rew_mean",
    "Episode Length": "rollout/ep_len_mean",
    "Loss": "train/loss",
    "Entropy": "train/entropy_loss",
    "Value Function": "train/value_loss",
    "Policy Loss": "train/policy_gradient_loss",
    # "Gradient Norms": "train/grad_norm"
}

INVERTED_TAGS = {
    'eval/mean_reward': "Evaluation Episode Reward",
    # 'eval/mean_ep_length': "Evaluation Episode Length",
    "rollout/ep_rew_mean": "Episode Reward",
    # "rollout/ep_len_mean": "Episode Length",
    # "train/loss": "Loss",
    # "train/entropy_loss": "Entropy",
    # "train/value_loss": "Value Function",
    # "train/policy_gradient_loss": "Policy Loss"
    # "train/grad_norm": "Gradient Norms"  # Uncomment if needed
}

ACTION_DISTRIBUTION_TAGS = {
    'eval/action_count_0': 0,
    'eval/action_count_1': 1,
    'eval/action_count_2': 2,
    'eval/action_count_3': 3,
    'eval/action_count_4': 4,
    'eval/action_count_5': 5,
    'eval/action_count_6': 6,
    'eval/action_count_7': 7,
    'eval/action_count_8': 8,
    'eval/action_count_9': 9,
    'eval/action_count_10': 10,
    'eval/action_count_11': 11,
    'eval/action_count_12': 12,
}

# eval/mean_ep_length
# eval/mean_reward
# train/approx_kl
# train/clip_fraction
# train/clip_range
# train/explained_variance
# train/learning_rate

# train/loss
# train/entropy_loss
# train/policy_gradient_loss
# train/value_loss
# rollout/ep_len_mean
# rollout/ep_rew_mean

# SHARED_DIR = "experiments/combined_tuned_params/training_logs/level_change_random/"
FILE_NAME = "1000_tuned_exp_params"
SHARED_DIR = f"/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/training_logs/experiments/{FILE_NAME}/training_logs/level_change_random/"

SUP_LOG_DIR = f"{SHARED_DIR}supervised/amalgam/PPO_20M_agent_"

# SUP_EXP_LOG_DIR = f"experiments/tuned_params/training_logs/level_change_random/supervised/expert_distance/PPO_20M_agent_"
# SUP_EXP_LOG_DIR = f"{SHARED_DIR}supervised/expert_distance/PPO_20M_agent_"
# SUP_NONEXP_LOG_DIR = f"{SHARED_DIR}supervised/nonexpert_distance/PPO_20M_agent_"
UNSUP_LOG_DIR = f"{SHARED_DIR}unsupervised/PPO_20M_agent_"
# UNSUP_BC_LOG_DIR = "level_change_random/unsupervised_bc_tuning/unsupervised_PPO_10000k_"
AGENT_START_INDEX = 0
NUM_AGENTS = 5
TRAINING = True
# AGENT_INFO = "comparison_of_all_PPO_20M_"
# AGENT_INFO = "sup_exp_nonexp_5000k_"
AGENT_INFO = "unsup_sup_amalagam_PPO_20M_"
# LOG_DIR += AGENT_INFO

PLOT_PATH = f"{SHARED_DIR}/training_plots/"

if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH, exist_ok=True)

LOWESS_FRAC = 0.1

PLOT_PATH = f"{PLOT_PATH}{AGENT_INFO}"

def extract_tfevents_data(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()

    # Extract all scalar data
    scalar_data = {}
    for tag in ea.Tags()['scalars']:
        try:
            scalar_data[tag] = pd.DataFrame(ea.Scalars(tag))
            scalar_data[tag]['value'] = scalar_data[tag]['value'].astype(float)  # Convert to float for Plotly
        except KeyError as e:
            print(f"KeyError: {e} - The tag '{tag}' was not found in the logs.")

    return scalar_data

def find_run_subdir(directory):
    for entry in os.listdir(directory):  # List all items in the directory
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path) and entry.startswith("run"):  # Check if it's a directory and starts with "run"
            return full_path  # Return the first match (modify as needed)

    return None  # Return None if no match is found

# Function to load data from multiple runs
def load_multiple_runs(log_dirs):
    all_runs = {}
    for run_name, log_dir in log_dirs.items():
        print(log_dir)

        # if run directory exists:
        # any(name.startswith("run_") and os.path.isdir(os.path.join(log_dir, name)) for name in os.listdir(log_dir))
        # then update the dir to include that at the end? 
        # if more than one exists I need to stack them together though??
        log_dir = find_run_subdir(log_dir)
        scalar_data = extract_tfevents_data(log_dir)
        for tag, df in scalar_data.items():
            df['run'] = run_name  # Add a column for the run name
            if tag not in all_runs:
                all_runs[tag] = df
            else:
                all_runs[tag] = pd.concat([all_runs[tag], df], ignore_index=True)
    # print(all_runs)
    return all_runs

def plot_mean_and_std(combined_data, plot_name, x_axis_name, plot_path):

    combined_df = pd.concat(combined_data, keys=['0', '1', '2'], names=['run', 'step'])

    # Reset index to make 'run' a column instead of an index
    combined_df = combined_df.reset_index(level='run')

    # Group by the time steps and compute the mean
    mean_df = combined_df.groupby('step').mean().reset_index()
    mean_df['std'] = combined_df.groupby('step')['value'].std().reset_index(drop=True)

    fig = px.line(mean_df, x='step', y='value', error_y='std', title='Mean Across Multiple Runs with Error Bars')
    fig.show()



    # Calculate mean and standard deviation
    tag_name = TAGS[plot_name]
    df = combined_data[tag_name]
    mean = df.groupby("Step")["Value"].mean()
    std = df.groupby("Step")["Value"].std()

    # plot_name = INVERTED_TAGS[tag_name]

    plot_title = f"{plot_name} vs. {x_axis_name}"

    # Plot
    fig = go.Figure()

    # Add mean line
    fig.add_trace(go.Scatter(x=mean.index, y=mean,
                            mode='lines', name=f'Mean {plot_name}'))

    # Add standard deviation band
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean + std,
        mode='lines', name='Upper Bound', line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean - std,
        mode='lines', name='Lower Bound', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
        showlegend=False
    ))

    fig.update_layout(title=plot_title, xaxis_title=x_axis_name, yaxis_title=plot_name)
    plot_name_for_path = plot_name.replace(" ", "_")
    fig.write_image(f"{plot_path}_{plot_name_for_path}_vs_{x_axis_name}_(avg).png")
    fig.write_html(f"{plot_path}_{plot_name_for_path}_vs_{x_axis_name}_(avg).html")
    fig.show()


def plot_multiple_traces(combined_data, plot_name, x_axis_name, plot_path):
    tag_name = TAGS[plot_name]

    plot_title = f"{plot_name} vs. {x_axis_name}"
    fig = px.line(combined_data[tag_name], x='value', y=tag_name, color="Run",
              title=plot_title)

    plot_name_for_path = plot_name.replace(" ", "_")
    fig.write_image(f"{plot_path}_{plot_name_for_path}_vs_{x_axis_name}.png")
    fig.write_html(f"{plot_path}_{plot_name_for_path}_vs_{x_axis_name}.html")
    fig.show()


def plot_mean_with_variance(all_runs_data, metrics):
    for metric in metrics:
        if metric in all_runs_data:
            df = all_runs_data[metric]
            # Group by step to calculate mean and standard deviation
            mean_df = df.groupby('step').agg({'value': 'mean', 'run': 'count'}).reset_index()
            std_df = df.groupby('step').agg({'value': 'std'}).reset_index()
            mean_df['std'] = std_df['value']
            
            # Plot with error bars
            fig = px.line(mean_df, x='step', y='value', error_y='std', title=f'Mean and Variance of {metric}')
            fig.update_layout(yaxis_title=metric)
            fig.show()


# Function to plot the mean with shaded variance for each metric
def plot_mean_with_shaded_variance_subplots(agents, metrics, plot_path=PLOT_PATH):

    num_agents = len(agents)
    num_cols = 2
    num_rows = int(num_agents/num_cols)

    for metric in metrics:
        metric_name = metrics[metric]

        fig = make_subplots(num_rows, num_cols, subplot_titles=list(agents.keys()))
        # fig = go.Figure()

        row = 1
        col = 1

        for agent in agents:

            line_color = agents[agent]['line_colour']
            fill_color = agents[agent]['fill_colour']

            all_runs_data = agents[agent]['data']
            if metric in all_runs_data:
                df = all_runs_data[metric]
                print(df.head())
                # Group by step to calculate mean and standard deviation
                mean_df = df.groupby('step').agg({'value': 'mean'}).reset_index()
                std_df = df.groupby('step').agg({'value': 'std'}).reset_index()
                mean_df['std'] = std_df['value']
                
                # Create the shaded area (mean ± std)
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'], 
                        y=mean_df['value'] + mean_df['std'], 
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ), row=row, col=col)
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'], 
                        y=mean_df['value'] - mean_df['std'], 
                        mode='lines',
                        fill='tonexty',  # Fill area between the lines
                        fillcolor=fill_color,
                        line=dict(width=0),
                        showlegend=False
                    ), row=row, col=col)
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'],
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    ), row=row, col=col)

            col += 1

            if col > num_cols:
                col = 1
                row += 1

        fig.update_layout(title=f'Mean and Variance of {metric_name}', xaxis_title='Step', yaxis_title=metric_name)

        plot_name_for_path = metric_name.replace(" ", "_")
        plot_name_for_path += "_subplots"
        fig.write_image(f"{plot_path}{plot_name_for_path}.png")
        fig.write_html(f"{plot_path}{plot_name_for_path}.html")

# Function to plot the mean with shaded variance for each metric
def plot_mean_with_shaded_variance(agents, metrics, plot_path=PLOT_PATH):
    for metric in metrics:
        metric_name = metrics[metric]
        fig = go.Figure()

        for agent in agents:

            line_color = agents[agent]['line_colour']
            fill_color = agents[agent]['fill_colour'] #'rgba(0, 0, 255, 0.2)'  # RGBA for blue with alpha=0.2 for transparency

            all_runs_data = agents[agent]['data']
            if metric in all_runs_data:
                df = all_runs_data[metric]
                print(df.head())
                # Group by step to calculate mean and standard deviation
                mean_df = df.groupby('step').agg({'value': 'mean'}).reset_index()
                std_df = df.groupby('step').agg({'value': 'std'}).reset_index()
                mean_df['std'] = std_df['value']
                
                # Create the shaded area (mean ± std)
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'], 
                        y=mean_df['value'] + mean_df['std'], 
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'], 
                        y=mean_df['value'] - mean_df['std'], 
                        mode='lines',
                        fill='tonexty',  # Fill area between the lines
                        fillcolor=fill_color,
                        line=dict(width=0),
                        showlegend=False
                    ))
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'],
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    ))

                fig.update_layout(title=f'Mean and Variance of {metric_name}', xaxis_title='Step', yaxis_title=metric_name)

        plot_name_for_path = metric_name.replace(" ", "_")
        fig.write_image(f"{plot_path}{plot_name_for_path}.png")
        fig.write_html(f"{plot_path}{plot_name_for_path}.html")
        # fig.show()


def plot_lowess_subplots(agents, metrics, plot_path=PLOT_PATH, use_mean=True, with_var=True):

    num_agents = len(agents)
    num_cols = 2
    num_rows = int(num_agents/num_cols)

    for metric in metrics:
        metric_name = metrics[metric]

        fig = make_subplots(num_rows, num_cols, subplot_titles=list(agents.keys()))
        # fig = go.Figure()

        row = 1
        col = 1

        for agent in agents:

            line_color = agents[agent]['line_colour']
            fill_color = agents[agent]['fill_colour']

            all_runs_data = agents[agent]['data']
            if metric in all_runs_data:
                df = all_runs_data[metric]
                print(df.head())
                # Group by step to calculate mean and standard deviation
                mean_df = df.groupby('step').agg({'value': 'mean'}).reset_index()
                std_df = df.groupby('step').agg({'value': 'std'}).reset_index()
                mean_df['std'] = std_df['value']
                
                if use_mean:
                    loess_result = lowess(mean_df['value'], mean_df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span
                else:
                    loess_result = lowess(df['value'], df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span
                
                # Extract smoothed values
                x_smooth, y_smooth = loess_result[:, 0], loess_result[:, 1]

                if with_var:
                    # Apply LOESS smoothing
                    loess_result = lowess(mean_df['std'], mean_df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span

                    # Extract smoothed values
                    std_x_smooth, std_y_smooth = loess_result[:, 0], loess_result[:, 1]

                    # Create the shaded area (mean ± std)
                    fig.add_trace(
                        go.Scatter(
                            x=std_x_smooth, 
                            y=y_smooth + std_y_smooth, 
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ), row=row, col=col)
                    fig.add_trace(
                        go.Scatter(
                            x=std_x_smooth, 
                            y=y_smooth - std_y_smooth, 
                            mode='lines',
                            fill='tonexty',  # Fill area between the lines
                            fillcolor=fill_color,
                            line=dict(width=0),
                            showlegend=False
                        ), row=row, col=col)
                fig.add_trace(
                    go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    ), row=row, col=col)

            col += 1

            if col > num_cols:
                col = 1
                row += 1


        plot_title = "LOWESS "

        plot_title += "Mean " if use_mean else ""
        plot_title += "and Variance " if with_var else ""

        plot_title += f"of {metric_name}"

        fig.update_layout(title=plot_title, xaxis_title='Step', yaxis_title=metric_name)

        plot_name_for_path = metric_name.replace(" ", "_")

        plot_name_for_path += "_mean" if use_mean else ""
        plot_name_for_path += "_lowess"
        plot_name_for_path += "_only" if not with_var else ""

        plot_name_for_path += "_subplots"
        fig.write_image(f"{plot_path}{plot_name_for_path}.png")
        fig.write_html(f"{plot_path}{plot_name_for_path}.html")


def plot_lowess(agents, metrics, plot_path=PLOT_PATH, use_mean=True, with_scatter=False, with_var=True):
    for metric in metrics:
        metric_name = metrics[metric]
        fig = go.Figure()

        for agent in agents:

            line_color = agents[agent]['line_colour']
            fill_color = agents[agent]['fill_colour'] #'rgba(0, 0, 255, 0.2)'  # RGBA for blue with alpha=0.2 for transparency

            all_runs_data = agents[agent]['data']
            if metric in all_runs_data:
                df = all_runs_data[metric]
                print(df.head())
                # Group by step to calculate mean and standard deviation
                mean_df = df.groupby('step').agg({'value': 'mean'}).reset_index()
                std_df = df.groupby('step').agg({'value': 'std'}).reset_index()
                mean_df['std'] = std_df['value']
                
                if use_mean:
                    loess_result = lowess(mean_df['value'], mean_df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span
                else:
                    loess_result = lowess(df['value'], df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span
                
                # Extract smoothed values
                x_smooth, y_smooth = loess_result[:, 0], loess_result[:, 1]

                if with_var:
                    # Apply LOESS smoothing
                    loess_result = lowess(mean_df['std'], mean_df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span

                    # Extract smoothed values
                    std_x_smooth, std_y_smooth = loess_result[:, 0], loess_result[:, 1]

                    if not use_mean:
                        df_mean = pd.DataFrame({
                            "x": x_smooth,
                            "y": y_smooth
                        })

                        df_std = pd.DataFrame({
                            "x": std_x_smooth,
                            "std": std_y_smooth
                        })

                        df_std_interp = pd.merge_asof(
                            df_mean.sort_values("x"),
                            df_std.sort_values("x"),
                            on="x"
                        )

                        std_x_smooth = df_std_interp["x"]
                        std_y_smooth = df_std_interp["y"]

                    # Create the shaded area (mean ± std)
                    fig.add_trace(
                        go.Scatter(
                            x=std_x_smooth, 
                            y=y_smooth + std_y_smooth, 
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                    fig.add_trace(
                        go.Scatter(
                            x=std_x_smooth, 
                            y=y_smooth - std_y_smooth, 
                            mode='lines',
                            fill='tonexty',  # Fill area between the lines
                            fillcolor=fill_color,
                            line=dict(width=0),
                            showlegend=False
                        ))
                
                fig.add_trace(
                    go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    ))

                if use_mean and with_scatter:
                    fig.add_trace(
                        go.Scatter(
                            x=mean_df['step'],
                            y=mean_df['value'],
                            mode='markers',
                            name=f"{agent} - mean values",
                            line=dict(color=fill_color)
                        ))

                plot_title = "LOWESS "

                plot_title += "Mean " if use_mean else ""
                plot_title += "and Variance " if with_var else ""

                plot_title += f"of {metric_name}"

                fig.update_layout(title=plot_title, xaxis_title='Step', yaxis_title=metric_name)

        plot_name_for_path = metric_name.replace(" ", "_")

        plot_name_for_path += "_mean" if use_mean else ""
        plot_name_for_path += "_lowess"
        plot_name_for_path += "_only" if not with_var else ""
        plot_name_for_path += "_with_scatter" if use_mean and with_scatter else ""
        
        fig.write_image(f"{plot_path}{plot_name_for_path}.png")
        fig.write_html(f"{plot_path}{plot_name_for_path}.html")


def plot_lowess_gradient(agents, metrics, plot_path=PLOT_PATH, use_mean=True, instantaneous=True, rolling=False, window_size=10):
    for metric in metrics:
        metric_name = metrics[metric]
        fig = go.Figure()

        for agent in agents:

            line_color = agents[agent]['line_colour']
            fill_color = agents[agent]['fill_colour'] #'rgba(0, 0, 255, 0.2)'  # RGBA for blue with alpha=0.2 for transparency

            all_runs_data = agents[agent]['data']
            if metric in all_runs_data:
                df = all_runs_data[metric]
                print(df.head())
                # Group by step to calculate mean and standard deviation
                mean_df = df.groupby('step').agg({'value': 'mean'}).reset_index()
                std_df = df.groupby('step').agg({'value': 'std'}).reset_index()
                mean_df['std'] = std_df['value']
                
                if use_mean:
                    loess_result = lowess(mean_df['value'], mean_df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span
                else:
                    loess_result = lowess(df['value'], df['step'], frac=LOWESS_FRAC)  # frac determines the smoothing span
                
                # Extract smoothed values
                x_smooth, y_smooth = loess_result[:, 0], loess_result[:, 1]

                if instantaneous:
                    gradients = np.gradient(y_smooth, x_smooth)
                
                elif rolling:
                    gradient = np.gradient(y_smooth, x_smooth)#, 1, 1000)
                    # window_size = 10

                    gradients = pd.Series(gradient).rolling(window=window_size, center=True, min_periods=1).mean()

                    # # Ensure x_smooth and rolling_gradient have the same length
                    # assert len(x_smooth) == len(gradients), "Mismatch in x_smooth and rolling_gradient lengths"

                    # x_smooth = []
                    # gradients = []

                    # for x in range(int(len(x_smooth)/window_size)):
                    #     mid_x = (window_size - x)/2
                    #     x_smooth.append(mid_x)
                    #     if int(x+window_size) > len(x_smooth):
                    #         break
                    #     gradient = np.mean(np.gradient(y_smooth[x:int(x+window_size)], x_smooth[x:int(x+window_size)]))
                    #     gradients.append(gradient)
                        # gradients = pd.Series(gradient).rolling(window=window_size, center=True).mean()

                else:
                    gradients = []

                    for i in range(len(x_smooth) - 1):
                        gradient = np.mean(np.gradient(y_smooth[i:], x_smooth[i:]))
                        gradients.append(gradient)

                # gradient = np.gradient(y_smooth, x_smooth)

                fig.add_trace(
                    go.Scatter(
                        x=x_smooth,
                        y=gradients,
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    ))

                plot_title = "LOWESS "
                plot_title += "Instantaneous " if instantaneous else ""
                plot_title += "Rolling " if rolling else ""

                plot_title += f"Gradient of "
                plot_title += f"Mean " if use_mean else ""
                plot_title += f"{metric_name}"

                fig.update_layout(title=plot_title, xaxis_title='Step', yaxis_title=metric_name)

        plot_name_for_path = metric_name.replace(" ", "_")

        plot_name_for_path += "_mean" if use_mean else ""
        plot_name_for_path += "_instantaneous" if instantaneous else ""
        plot_name_for_path += f"_rolling_{window_size}" if rolling else ""
        plot_name_for_path += "_gradient_lowess"

        fig.write_image(f"{plot_path}{plot_name_for_path}.png")
        fig.write_html(f"{plot_path}{plot_name_for_path}.html")


def plot_mean_var_actions(agents, plot_path=PLOT_PATH, action_mapping=ACTION_MAPPING, action_distribution_tags=ACTION_DISTRIBUTION_TAGS):
    for action_tag, action_index in action_distribution_tags.items():
        fig = go.Figure()

        # Extract evaluation steps and runs once to avoid redundant looping
        eval_steps = next(iter(agents.values()))['data']['eval/mean_reward'][['step', 'run']].drop_duplicates()

        for agent, agent_data in agents.items():
            line_color = agent_data['line_colour']
            fill_color = agent_data['fill_colour']

            all_runs_data = agent_data['data']

            if action_tag in all_runs_data:
                df = all_runs_data[action_tag]

                # Merge with eval_steps to ensure all steps/runs exist, then fill missing values
                df = eval_steps.merge(df, on=['step', 'run'], how='left').fillna(0)

                # Group by step to calculate sum, mean, and std
                grouped = df.groupby('step')['value']
                mean_df = grouped.mean().reset_index()
                std_df = grouped.std().reset_index()
                sum_df = grouped.sum().reset_index()

                # Merge mean and std into one dataframe
                mean_df['std'] = std_df['value']
                mean_df['sum'] = sum_df['value']

                # Create shaded area (mean ± std)
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'] + mean_df['std'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'] - mean_df['std'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=fill_color,
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'],
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    )
                )

        # Update layout
        fig.update_layout(
            title=f'Mean and Variance of Action {action_index} ({action_mapping[action_index]}) Use in Evaluations',
            xaxis_title="Step",
            yaxis_title=f"Action {action_index} count"
        )

        # Save the plot
        plot_name = f"count_action_{action_index}_{action_mapping[action_index]}"
        fig.write_image(f"{plot_path}{plot_name}.png")
        fig.write_html(f"{plot_path}{plot_name}.html")


def plot_one_plot_mean_var_actions(agents, plot_path=PLOT_PATH, action_mapping=ACTION_MAPPING, action_distribution_tags=ACTION_DISTRIBUTION_TAGS):
    # Define grid dimensions
    rows, cols = 4, 4

    # Generate labels efficiently using list comprehension
    labels = [
        f"Action {i} ({action_mapping[i]})" if i < 13 else None
        for i in range(rows * cols)
    ]

    # Create subplot figure
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)

    # Extract evaluation steps and runs once to avoid redundant looping
    eval_steps = next(iter(agents.values()))['data']['eval/mean_reward'][['step', 'run']].drop_duplicates()

    # Efficiently loop through actions and agents
    for action_tag, action_index in action_distribution_tags.items():
        row, col = divmod(action_index, rows)  # Efficient row-column mapping
        row += 1
        col += 1

        for agent, agent_data in agents.items():
            line_color = agent_data['line_colour']
            fill_color = agent_data['fill_colour']

            all_runs_data = agent_data['data']

            if action_tag in all_runs_data:
                df = all_runs_data[action_tag]

                # Merge with eval_steps to ensure all steps/runs exist, then fill missing values
                df = eval_steps.merge(df, on=['step', 'run'], how='left').fillna(0)

                # Group by step to calculate sum, mean, and std
                grouped = df.groupby('step')['value']
                mean_df = grouped.mean().reset_index()
                std_df = grouped.std().reset_index()
                sum_df = grouped.sum().reset_index()

                # Merge mean and std into one dataframe
                mean_df['std'] = std_df['value']
                mean_df['sum'] = sum_df['value']

                # Create shaded area (mean ± std)
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'] + mean_df['std'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'] - mean_df['std'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=fill_color,
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_df['step'],
                        y=mean_df['value'],
                        mode='lines',
                        name=agent,
                        line=dict(color=line_color)
                    ),
                    row=row,
                    col=col
                )

    # Update layout and save
    fig.update_layout(
        title="Mean and Variance of Action Uses in Evaluations",
        xaxis_title="Step",
        yaxis_title="Action Count"
    )

    plot_name = "all_action_eval_uses"
    fig.write_image(f"{plot_path}{plot_name}.png")
    fig.write_html(f"{plot_path}{plot_name}.html")


def plot_one_plot_cumulative_actions(agents, plot_path=PLOT_PATH, action_mapping=ACTION_MAPPING, action_distribution_tags=ACTION_DISTRIBUTION_TAGS):
    # Define grid dimensions
    rows, cols = 4, 4

    # Generate labels efficiently using list comprehension
    labels = [
        f"Action {i} ({action_mapping[i]})" if i < 13 else None
        for i in range(rows * cols)
    ]

    # Create subplot figure
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels)

    # Extract evaluation steps and runs once to avoid redundant looping
    eval_steps = next(iter(agents.values()))['data']['eval/mean_reward'][['step', 'run']].drop_duplicates()

    # Efficiently loop through actions and agents
    for action_tag, action_index in action_distribution_tags.items():
        row, col = divmod(action_index, rows)  # Efficient row-column mapping
        # print(row, col)
        row += 1
        col += 1

        # if col > cols:
        #     col = 1
        #     row += 1

        for agent, agent_data in agents.items():
            all_runs_data = agent_data['data']

            if action_tag in all_runs_data:
                df = all_runs_data[action_tag]

                # Merge with eval_steps to ensure all steps/runs exist, then fill missing values
                df = eval_steps.merge(df, on=['step', 'run'], how='left').fillna(0)

                # Group by run and sum values
                sum_arr = df.groupby('run')['value'].sum().to_numpy()

                # Compute mean and standard deviation
                mean, std = sum_arr.mean(), sum_arr.std()

                # Add trace to subplot
                fig.add_trace(
                    go.Bar(
                        x=[action_index],
                        y=[mean],
                        error_y=dict(type='data', array=[std], visible=True),
                        name=agent
                    ),
                    row=row,
                    col=col
                )

    # Update layout and save
    fig.update_layout(
        title="Cumulative Action Count in Evaluations",
        xaxis_title="Agent Index",
        yaxis_title="Action Count"
    )

    plot_name = "all_action_eval_cumulative_count"
    fig.write_image(f"{plot_path}{plot_name}.png")
    fig.write_html(f"{plot_path}{plot_name}.html")


def plot_bar_cumulative_actions(agents, plot_path=PLOT_PATH, action_mapping=ACTION_MAPPING, action_distribution_tags=ACTION_DISTRIBUTION_TAGS):
    fig = go.Figure()

    for agent, agent_data in agents.items():
        all_runs_data = agent_data['data']

        agent_action_list = []

        eval_steps = all_runs_data['eval/mean_reward'][['step', 'run']].drop_duplicates()

        for action_tag, action_index in action_distribution_tags.items():
            if action_tag in all_runs_data:
                df = all_runs_data[action_tag]

                # Ensure all steps and runs exist
                df = eval_steps.merge(df, on=['step', 'run'], how='left').fillna(0)

                # Group and calculate stats
                sum_arr = df.groupby('run')['value'].sum().to_numpy()
                mean, std = sum_arr.mean(), sum_arr.std()

                # Store values
                agent_action_list.append((action_mapping[action_index], mean, std))  # Use labels

        # Convert list to DataFrame
        agent_action_df = pd.DataFrame(agent_action_list, columns=["Action", "Mean", "Std"])

        # Add trace using labels
        fig.add_trace(
            go.Bar(
                x=agent_action_df['Action'],  # Use action labels
                y=agent_action_df['Mean'],
                error_y=dict(type='data', array=agent_action_df['Std'], visible=True),
                name=agent
            )
        )

    # Update layout
    fig.update_layout(
        title="Cumulative Action Count in Evaluations",
        xaxis_title="Action Label",  # Updated x-axis title
        yaxis_title="Action Count",
        barmode="group"
    )

    # Save plot
    plot_name = "all_action_eval_cumulative_count_bars"
    fig.write_image(f"{plot_path}{plot_name}.png")
    fig.write_html(f"{plot_path}{plot_name}.html")

def plot_bar_cumulative_actions_no_error_bars(agents, plot_path=PLOT_PATH, action_mapping=ACTION_MAPPING, action_distribution_tags=ACTION_DISTRIBUTION_TAGS):
    fig = go.Figure()

    for agent, agent_data in agents.items():
        all_runs_data = agent_data['data']

        agent_action_list = []

        eval_steps = all_runs_data['eval/mean_reward'][['step', 'run']].drop_duplicates()

        for action_tag, action_index in action_distribution_tags.items():
            if action_tag in all_runs_data:
                df = all_runs_data[action_tag]

                # Ensure all steps and runs exist
                df = eval_steps.merge(df, on=['step', 'run'], how='left').fillna(0)

                # Group and calculate stats
                sum_arr = df.groupby('run')['value'].sum().to_numpy()
                mean, std = sum_arr.mean(), sum_arr.std()

                # Store values
                agent_action_list.append((action_mapping[action_index], mean, std))  # Use labels

        # Convert list to DataFrame
        agent_action_df = pd.DataFrame(agent_action_list, columns=["Action", "Mean", "Std"])

        # Add trace using labels
        fig.add_trace(
            go.Bar(
                x=agent_action_df['Action'],  # Use action labels
                y=agent_action_df['Mean'],
                # error_y=dict(type='data', array=agent_action_df['Std'], visible=True),
                name=agent
            )
        )

    # Update layout
    fig.update_layout(
        title="Cumulative Action Count in Evaluations",
        xaxis_title="Action Label",  # Updated x-axis title
        yaxis_title="Action Count",
        barmode="group"
    )

    # Save plot
    plot_name = "all_action_eval_cumulative_count_bars_no_error_bars"
    fig.write_image(f"{plot_path}{plot_name}.png")
    fig.write_html(f"{plot_path}{plot_name}.html")


def main():
    
    # print(list(INVERTED_TAGS.keys()))
    sup_log_dirs = {}
    unsup_log_dirs = {}
    # exp_log_dirs = {}
    # nonexp_log_dirs = {}

    for i in range(NUM_AGENTS):
        agent_index = i + AGENT_START_INDEX
        sup_log_dirs[agent_index] = SUP_LOG_DIR + str(agent_index)
        # exp_log_dirs[agent_index] = SUP_EXP_LOG_DIR + str(agent_index)
        # nonexp_log_dirs[agent_index] = SUP_NONEXP_LOG_DIR + str(agent_index)
        unsup_log_dirs[agent_index] = UNSUP_LOG_DIR + str(agent_index)

    # Load data from all runs
    sup_all_runs_data = load_multiple_runs(sup_log_dirs)
    # sup_exp_all_runs_data = load_multiple_runs(exp_log_dirs)
    # sup_nonexp_all_runs_data = load_multiple_runs(nonexp_log_dirs)
    unsup_all_runs_data = load_multiple_runs(unsup_log_dirs)
    # unsup_bc_all_runs_data = load_multiple_runs(unsup_bc_log_dirs)

    fill_colours = []

    for colour in COLOUR_SCHEME[:5]:
        fill_colour = 'rgba' + colour[3:-1] + ', 0.2)'
        fill_colours.append(fill_colour)

    agents = {"Unsupervised": {"data": unsup_all_runs_data, "line_colour": COLOUR_SCHEME[0], "fill_colour": fill_colours[0]},
              "Supervised<br>(amalgam)": {"data": sup_all_runs_data, "line_colour": COLOUR_SCHEME[1], 'fill_colour': fill_colours[1]},
            }

    # agents = {"Unsupervised": {"data": unsup_all_runs_data, "line_colour": COLOUR_SCHEME[0], "fill_colour": fill_colours[0]},
    #           "Supervised<br>(amalgam)": {"data": sup_all_runs_data, "line_colour": COLOUR_SCHEME[1], 'fill_colour': fill_colours[1]},
    #           "Supervised<br>(expert)": {"data": sup_exp_all_runs_data, "line_colour": COLOUR_SCHEME[2], 'fill_colour': fill_colours[2]},
    #           "Supervised<br>(nonexpert)": {"data": sup_nonexp_all_runs_data, "line_colour": COLOUR_SCHEME[3], 'fill_colour': fill_colours[3]}}

    # Plot mean with variance for each metric
    plot_mean_with_shaded_variance(agents, INVERTED_TAGS)
    plot_mean_with_shaded_variance_subplots(agents, INVERTED_TAGS)

    plot_lowess(agents, INVERTED_TAGS, use_mean=True, with_scatter=False, with_var=True)
    # plot_lowess(agents, INVERTED_TAGS, use_mean=True, with_scatter=False, with_var=False)
    # plot_lowess(agents, INVERTED_TAGS, use_mean=True, with_scatter=True, with_var=False)
    # plot_lowess(agents, INVERTED_TAGS, use_mean=False, with_scatter=False, with_var=False)

    # plot_lowess_subplots(agents, INVERTED_TAGS, use_mean=False, with_var=False)
    # plot_lowess_subplots(agents, INVERTED_TAGS, use_mean=True, with_var=False)
    plot_lowess_subplots(agents, INVERTED_TAGS, use_mean=True, with_var=True)

    # for boolean in [True, False]:
    #     plot_lowess_gradient(agents, INVERTED_TAGS, use_mean=True, instantaneous=boolean, rolling=(not boolean), window_size=5000)

    # plot_lowess_gradient(agents, INVERTED_TAGS, use_mean=True, instantaneous=False, rolling=False)
    plot_lowess_gradient(agents, INVERTED_TAGS, use_mean=True, instantaneous=False, rolling=True, window_size=5000)

    plot_bar_cumulative_actions(agents)
    # plot_bar_cumulative_actions_no_error_bars(agents)
    plot_one_plot_cumulative_actions(agents)
    plot_mean_var_actions(agents)
    plot_one_plot_mean_var_actions(agents)

    # # Load the data from all runs
    # combined_data = load_multiple_runs(log_dirs)

    # # # action distribution vs episodes
    # # fig = px.line(data[tags["Action Distribution"]], x="Step", y="Value", title="Action Distribution vs. Episodes")
    # # fig.show()


if __name__=="__main__":
    main()

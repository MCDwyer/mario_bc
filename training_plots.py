import tensorflow as tf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Specify the tags you want to extract
tags = {
    "Episode Reward": "rollout/ep_rew_mean",
    "Loss": "train/loss",
    "Entropy": "train/entropy_loss",
    "Value Function": "train/value_loss",
    "Policy Loss": "train/policy_loss",
    "Gradient Norms": "train/grad_norm"
}

LOG_DIR = ""
AGENT_START_INDEX = 0
NUM_AGENTS = 3
TRAINING = True
AGENT_INFO = "1000k_"
LOG_DIR += AGENT_INFO

if TRAINING:
    PLOT_PATH = f"training_plots/{AGENT_INFO}"
else:
    PLOT_PATH =  f"test_plots/{AGENT_INFO}"



def load_tensorboard_data(log_dir, tags):
    """
    Load data from TensorBoard logs.
    :param log_dir: Directory where the TensorBoard logs are stored.
    :param tags: List of tags to extract (e.g., 'rollout/ep_rew_mean', 'train/loss').
    :return: Dictionary of DataFrames keyed by tag names.
    """
    data = {tag: [] for tag in tags}

    for event in tf.compat.v1.train.summary_iterator(log_dir):
        for value in event.summary.value:
            if value.tag in tags:
                data[value.tag].append([event.step, value.simple_value])

    dataframes = {tag: pd.DataFrame(values, columns=['Step', 'Value']) for tag, values in data.items()}
    return dataframes


# Load data from multiple runs
def load_multiple_runs(log_dirs, tags):
    all_data = {tag: [] for tag in tags}
    
    for log_dir in log_dirs:
        data = load_tensorboard_data(log_dir, tags)
        for tag, df in data.items():
            df['Run'] = log_dir  # Add a column to distinguish runs
            all_data[tag].append(df)
    
    # Combine data for each tag into a single DataFrame
    combined_data = {tag: pd.concat(all_data[tag], ignore_index=True) for tag in tags}
    return combined_data


def plot_mean_and_std(combined_data, tag_name, x_axis_name, plot_path):
    # Calculate mean and standard deviation
    df = combined_data[tags[tag_name]]
    mean = df.groupby("Step")["Value"].mean()
    std = df.groupby("Step")["Value"].std()

    plot_title = f"{tag_name} vs. {x_axis_name}"

    # Plot
    fig = go.Figure()

    # Add mean line
    fig.add_trace(go.Scatter(x=mean.index, y=mean,
                            mode='lines', name=f'Mean {tag_name}'))

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

    fig.update_layout(title=plot_title, xaxis_title=x_axis_name, yaxis_title=tag_name)
    fig.write_image(f"{plot_path}_{tag_name}_vs_{x_axis_name}_(avg).png")
    fig.write_html(f"{plot_path}_{tag_name}_vs_{x_axis_name}_(avg).html")
    fig.show()


def plot_multiple_traces(combined_data, tag_name, x_axis_name, plot_path):
    plot_title = f"{tag_name} vs. {x_axis_name}"
    fig = px.line(combined_data[tags[tag_name]], x=x_axis_name, y=tag_name, color="Run",
              title=plot_title)

    fig.write_image(f"{plot_path}_{tag_name}_vs_{x_axis_name}.png")
    fig.write_html(f"{plot_path}_{tag_name}_vs_{x_axis_name}.html")
    fig.show()


def main():
    log_dirs = []

    for i in range(NUM_AGENTS):
        agent_index = i + AGENT_START_INDEX
        log_dirs = LOG_DIR + str(agent_index)

    # Load the data from all runs
    combined_data = load_multiple_runs(log_dirs, tags.values())

    # episode reward vs episodes
    plot_mean_and_std(combined_data, "Episode Reward", "Episodes", PLOT_PATH)
    plot_multiple_traces(combined_data, "Episode Reward", "Episodes", PLOT_PATH)


    # loss vs training steps
    plot_mean_and_std(combined_data, "Loss", "Training Steps", PLOT_PATH)
    plot_multiple_traces(combined_data, "Loss", "Training Steps", PLOT_PATH)

    # # action distribution vs episodes
    # fig = px.line(data[tags["Action Distribution"]], x="Step", y="Value", title="Action Distribution vs. Episodes")
    # fig.show()

    # entropy vs steps
    plot_mean_and_std(combined_data, "Entropy", "Training Steps", PLOT_PATH)
    plot_multiple_traces(combined_data, "Entropy", "Training Steps", PLOT_PATH)


    # value function vs steps
    plot_mean_and_std(combined_data, "Value Function", "Training Steps", PLOT_PATH)
    plot_multiple_traces(combined_data, "Value Function", "Training Steps", PLOT_PATH)


    # policy loss vs steps
    plot_mean_and_std(combined_data, "Policy Loss", "Training Steps", PLOT_PATH)
    plot_multiple_traces(combined_data, "Policy Loss", "Training Steps", PLOT_PATH)


    # gradient norms vs training steps
    plot_mean_and_std(combined_data, "Gradient Norms", "Training Steps", PLOT_PATH)
    plot_multiple_traces(combined_data, "Gradient Norms", "Training Steps", PLOT_PATH)

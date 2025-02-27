from stable_baselines3 import PPO, DQN, SAC
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from scipy.stats import mannwhitneyu, shapiro, levene, ttest_ind, ttest_rel, wilcoxon, kruskal, f_oneway
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pickle
import os
import json
import shutil
import copy
import pandas as pd

TIMESTEPS = 20000000
UNSUPERVISED = False
RETRAINING = False
MODEL_NAME = "PPO"
MODEL_CLASS = PPO
POLICY = "CnnPolicy"
TRAINING_DATA_NAME = "amalgam"
# TRAINING_DATA_NAME = "expert_distance"
# TRAINING_DATA_NAME = "nonexpert_distance"
LEVEL_CHANGE = "random"

TRAINING_FILEPATH = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/user_data_processed_for_bc/"
TRAINING_FILEPATH += TRAINING_DATA_NAME + "_bc_data.obj"
# LEVEL_CHANGE = "single_level_Level1-1"

# training_data_name = "expert_score"
# training_data_name = "nonexpert_score"

# training_data_name = "slower"
# training_data_name = "faster"

ALL_LEVELS = ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]
TEST_EPISODES = 100#0
AGENT_INDICES = list(range(5))
MAX_Y = 768

def evaluate_model(model, env, level, n_episodes=10):

    all_trajectories = []
    all_actions = []
    all_action_distributions = []
    all_dist_rewards = []
    all_score_rewards = []
    all_combined_rewards = []

    for i in range(n_episodes):
        trajectory = []
        action_distribution = np.zeros(13, dtype=int)
        actions = []
        total_dist_reward = 0
        total_score_reward = 0
        total_combined_reward = 0
        total_reward = 0

        step = 0

        prev_position = 40
        prev_score = 0
        
        obs, _ = env.reset(options={"level": level})
        done = False

        state_change = False

        while not done:
            # step += 1

            # if step > 10:
            #     break
            action, _states = model.predict(obs)
            obs, reward, done, _, info = env.step(int(action))

            if done:
                break  # Exit immediately

            # action_distribution[int(action)] += 1
            np.add.at(action_distribution, int(action), 1)

            x = info["x_frame"]*256 + info["x_position_in_frame"]
            y = ((info["y_frame"]*256) + info["y_position_in_frame"])
            
            if info["player_state"] == 11 or level not in env.retro_env.statename or y > MAX_Y:# < -432:# or info["viewport_position"] > 1: #y < -432:# or info["player_dead"] != 32:# or y < -432:
                break

            trajectory.append([x, y])
            actions.append(action)

            # if state_change:
            #     dist_reward = 0
            #     score_reward = 0
            # else:
            #     dist_reward = (x - prev_position)
            #     score_reward = (info['score'] - prev_score)
            
            dist_reward = env.dist_reward
            score_reward = env.score_reward
            total_dist_reward += dist_reward
            total_score_reward += score_reward
            total_combined_reward += (dist_reward/2 + score_reward/2)

            total_reward += reward

            prev_position = x
            prev_score = info['score']

            if dist_reward < -2000:
                print(reward, score_reward, dist_reward)
                print(f"Why isn't this working!!!! player state = {info['player_state']}")

        all_trajectories.append(trajectory)
        all_actions.append(actions)
        all_action_distributions.append(action_distribution)
        all_dist_rewards.append(total_dist_reward)
        all_score_rewards.append(total_score_reward)
        all_combined_rewards.append(total_combined_reward)

        print(f"Episode {i} reward: {total_reward} - dist_reward: {total_dist_reward}")

    print(f"Distance Reward: {all_dist_rewards}")
    print(f"Score Reward: {all_score_rewards}")
    print(f"Combined Reward: {all_combined_rewards}")

    return all_trajectories, all_actions, all_action_distributions, all_dist_rewards, all_score_rewards, all_combined_rewards


# def evaluate_model(model, env, level, record_dir, n_episodes=10):
#     """
#     Evaluates the model by running it in the environment for n_episodes.
#     Returns the mean reward over these episodes.
#     """
#     episode_rewards = []
#     episode_trajectories = []
#     episode_info = []
#     episode_actions = []
#     total_actions = np.zeros(env.action_space.n)

#     for i in range(n_episodes):
#         record_path = record_dir + f"/episode_{i}/"
#         os.makedirs(record_path, exist_ok=True)

#         obs, _ = env.reset(options={"level": level, "record_option": record_path})
#         done = False
#         total_reward = 0.0
#         trajectory = []
#         scores = []
#         actions = np.zeros(env.action_space.n)

#         # print(env.retro_env.statename)
#         while not done:
#             action, _states = model.predict(obs)
#             obs, reward, done, _, info = env.step(int(action))

#             actions[int(action)] += 1

#             x = info["x_frame"]*256 + info["x_position_in_frame"]
#             y = ((info["y_frame"]*256) + info["y_position_in_frame"])
#             trajectory.append([x, y])
#             scores.append(info["score"])

#             total_reward += reward

#         print(f"Episode {i} reward: {total_reward}")
#         episode_rewards.append(total_reward)
#         episode_trajectories.append(trajectory)
#         episode_info.append({'score': np.max(np.array(scores)), 'max_dist': np.max(np.array(trajectory)[:, 0]), 'num_timesteps': len(trajectory)})
#         episode_actions.append(list(actions))
#         total_actions += actions

#     mean_reward = sum(episode_rewards) / n_episodes

#     return mean_reward, episode_rewards, episode_trajectories, episode_info, episode_actions

def get_results(model, env):
    mean_rewards = {}
    rewards = {}
    trajectories = {}

    for level in ALL_LEVELS:
        env.level = level

        mean_reward, rewards, episode_trajectories, episode_info = evaluate_model(model, env, level, ".", TEST_EPISODES)

        trajectories[level] = episode_trajectories
        mean_rewards[level] = mean_reward
        rewards[level] = rewards

    return mean_rewards, rewards, trajectories

def two_agent_statistical_tests(unsupervised_results, supervised_results):
    # Check the normality:
    # Run the Shapiro-Wilk test
    stat, p_value = shapiro(unsupervised_results)
    print(f"Unsupervised: Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")

    stat, p_value = shapiro(supervised_results)
    print(f"Supervised Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")

    # Perform Levene's Test
    stat, p_value = levene(unsupervised_results, supervised_results)

    print(f"Levene’s Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")        

    # Example: Assuming 'unsupervised_results' and 'supervised_results' are your performance metrics
    t_stat, p_value = ttest_ind(unsupervised_results, supervised_results)

    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # Example: Mann-Whitney U Test
    u_stat, p_value = mannwhitneyu(unsupervised_results, supervised_results)

    print(f"U-statistic: {u_stat}, P-value: {p_value}")

    # Perform paired t-test
    stat, p_value = ttest_rel(unsupervised_results, supervised_results)

    print(f"Paired t-test Statistic: {stat:.4f}, P-value: {p_value:.4f}")

    stat, p_value = wilcoxon(unsupervised_results, supervised_results)

    print(f"Wilcoxon Signed-Rank Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")

def four_agent_statistical_tests(agent_1, agent_2, agent_3, agent_4, mean_rewards=True):

    # Perform Levene's Test across four groups
    stat, p_value = levene(agent_1, agent_2, agent_3, agent_4)

    print(f"Levene’s Test Statistic (on all 4): {stat}, P-value: {p_value}")

    if mean_rewards:
        print("One way ANOVA (best on means):")
        f_stat, p_value = f_oneway(agent_1, agent_2, agent_3, agent_4)

        print(f"F-statistic: {f_stat}, P-value: {p_value}")

        # print("If ANOVA shows significant results, use Tukey’s HSD (Honestly Significant Difference) test to identify which groups are different.")
        # # Combine all rewards into a single array
        # all_rewards = agent_1 + agent_2 + agent_3 + agent_4
        # # Create labels for each group
        # groups = (['Agent1'] * len(agent_1) +
        #         ['Agent2'] * len(agent_2) +
        #         ['Agent3'] * len(agent_3) +
        #         ['Agent4'] * len(agent_4))

        # # Perform Tukey's HSD test
        # tukey = pairwise_tukeyhsd(endog=all_rewards, groups=groups, alpha=0.05)
        # print(tukey)
    else:
        print("Kruskal-Wallis H test")
        # Perform Kruskal-Wallis H Test
        h_stat, p_value = kruskal(agent_1, agent_2, agent_3, agent_4)

        print(f"Kruskal-Wallis H-statistic: {h_stat}, P-value: {p_value}")

    print()
    print()

def get_model_results(model_path, model_class, env):
    # eval_dir = model_path.split(".")[0] + "/evaluation/"

    os.makedirs(model_path.split(".")[0], exist_ok=True)

    model = model_class.load(model_path, env, verbose=1)

    # level_results = {}

    for level in ALL_LEVELS:
        # record_path = eval_dir + f"{level}/bk2_files" 
        # os.makedirs(record_path, exist_ok=True)

        print(f"\tEvaluating on {level} for {TEST_EPISODES} episodes.")

        # mean_reward, rewards, episode_trajectories, episode_info, episode_actions = evaluate_model(model, env, level, record_path, TEST_EPISODES)
        all_trajectories, all_actions, all_action_distributions, all_dist_rewards, all_score_rewards, all_combined_rewards = evaluate_model(model, env, level, TEST_EPISODES)
        
        # level_results[level] = {"trajectories": copy.deepcopy(all_trajectories),
        #                         "actions": copy.deepcopy(all_actions),
        #                         "rewards": copy.deepcopy(all_rewards),
        #                         "action_distributions": copy.deepcopy(all_action_distributions)}
        # print(f"\tMean reward is {mean_reward}, bk2 files stored in {record_path}.\n")
        # level_results[level] = {"mean_reward": mean_reward, "episode_data": []}

        df = pd.DataFrame.from_dict({"trajectories": all_trajectories,
                                "actions": all_actions,
                                "dist_rewards": all_dist_rewards,
                                "score_rewards": all_score_rewards,
                                "combined_rewards": all_combined_rewards,
                                "action_distributions": all_action_distributions})

        df_filename = model_path[:-4] + f"/{level}_dataframe.pkl"
        if not df.empty:
            print(f"{df_filename} being saved with {len(df.index)} rows of data.")
            df.to_pickle(df_filename) 
        else:
            print(f"{df_filename} not saved as dataframe is empty.")

    return 
    #     for i, episode_reward in enumerate(rewards):
    #         level_results[level]["episode_data"].append({
    #             "reward": episode_reward, 
    #             "trajectory": episode_trajectories[i], 
    #             "info": episode_info[i],
    #             "actions": episode_actions[i]})

    # json_filepath = eval_dir + "all_eval_info.json"

    # try:
    #     with open(json_filepath, "w") as json_file:
    #         print(f"All evaluation information being saved in {json_filepath}.\n")
    #         json.dump(level_results, json_file)
    # except: 
    #     backup_filepath = eval_dir + "all_eval_info.pkl"
    #     with open(backup_filepath, "wb") as pkl_file:
    #         print(f"All evaluation information being saved in {backup_filepath}.\n")
    #         pickle.dump(level_results, pkl_file)
    
    return level_results

def load_model_get_results(dir_path, string_timesteps, env, level_results, results, agent_index, level, record_path):
    model_path = dir_path + f"{string_timesteps}_{MODEL_NAME}_{agent_index}"

    model = MODEL_CLASS.load(model_path, env, verbose=1)

    mean_reward, rewards, episode_trajectories, episode_info = evaluate_model(model, env, level, record_path, TEST_EPISODES)

    level_results.append(mean_reward)

    if results is None:
        results = np.array(rewards)
    else:
        results = np.vstack((results, np.array(rewards)))
        
    return level_results, results, episode_trajectories, episode_info

def combine_level_data(results, agent_name, levels=ALL_LEVELS):
    all_means = None
    all_data = None

    for level in results:
        if level in levels:
            if all_means is None:
                all_means = np.array(results[level][agent_name]["means"])
            else:
                all_means = np.vstack((all_means, np.array(results[level][agent_name]["means"])))

            if all_data is None:
                all_data = np.array(results[level][agent_name]["all rewards"])
            else:
                all_data = np.vstack((all_data, np.array(results[level][agent_name]["all rewards"])))

    return all_means, all_data

def main():

    register(
            id='MarioEnv-v0',
            entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
        )

    env = gym.make('MarioEnv-v0')

    get_model_results(model_path, model_class, env)

    log_dir = f"training_logs/level_change_{LEVEL_CHANGE}/"

    env = gym.make('MarioEnv-v0')

    string_timesteps = f"{int(TIMESTEPS/1000)}k"

    # env.set_record_option("test_bk2s/.")

    env.level_change_type = "No Change"

    print(env.action_space)

    results_dict = {}

    dir_paths = {"unsupervised": f"{log_dir}unsupervised/", 
                 "supervised - amalgam": f"{log_dir}supervised/amalgam/", 
                 "supervised - expert": f"{log_dir}supervised/expert_distance/", 
                 "supervised - nonexpert": f"{log_dir}supervised/nonexpert_distance/"}

    dir_paths = {"supervised - amalgam": f"{log_dir}supervised/amalgam/"}

    trajectories = {}
    info = {}

    for level in ALL_LEVELS:
        env.level = level
        results_dict[level] = {}
        print(level)

        for agent_name in dir_paths:
            print(agent_name)
            results_dict[level][agent_name] = {"means": []}

            means_list = results_dict[level][agent_name]["means"]

            trajectories[agent_name] = {}
            info[agent_name] = {}

            results = None

            for agent_index in AGENT_INDICES:
                record_path = f"test_bk2s/{level}/{agent_name}/{agent_index}/"
                os.makedirs(record_path, exist_ok=True)
                record_path += "."
                trajectories[agent_name][agent_index] = {}
                means_list, results, trajectories, ep_info = load_model_get_results(dir_paths[agent_name], string_timesteps, env, means_list, results, agent_index, level, record_path)
                
                trajectories[agent_name][agent_index][level] = trajectories
                info[agent_name][agent_index] = {}
                info[agent_name][agent_index][level] = ep_info

            results_dict[level][agent_name]["all rewards"] = results
        
    with open('evaluations/trajectories.obj', 'wb') as handle:
        pickle.dump(trajectories, handle)

    with open('evaluations/infos.obj', 'wb') as handle:
        pickle.dump(info, handle)

    with open('evaluations/results.obj', 'wb') as handle:
        pickle.dump(results_dict, handle)

    combined_results = {}
    combined_results_training = {}
    combined_results_test = {}

    training_levels = ["Level1-1", "Level2-1", "Level4-1", "Level5-1", "Level6-1", "Level8-1"]
    test_levels = ["Level3-1", "Level7-1"]

    # run statistical tests on two agents
    for i, agent_name_1 in enumerate(dir_paths):
        for j, agent_name_2, in enumerate(dir_paths):        
            combined_results[agent_name_2] = {}
            combined_results[agent_name_2]["means"], combined_results[agent_name_2]["all rewards"] = combine_level_data(results, agent_name_2)

            combined_results_training[agent_name_2] = {}
            combined_results_training[agent_name_2]["means"], combined_results_training[agent_name_2]["all rewards"] = combine_level_data(results, agent_name_2, training_levels)

            combined_results_test[agent_name_2] = {}
            combined_results_test[agent_name_2]["means"], combined_results_test[agent_name_2]["all rewards"] = combine_level_data(results, agent_name_2, test_levels)

            if i == j:
                # skip as same agent
                continue

            print(f"2 agent tests on '{agent_name_1}' and '{agent_name_2}'")

            for level in results_dict:
                print(f"Per level tests on {level}")

                print("Tests on the mean rewards:")
                two_agent_statistical_tests(results_dict[level][agent_name_1]["means"], results_dict[level][agent_name_2]["means"])
                print()

                print("Tests on all the rewards:")
                two_agent_statistical_tests(results_dict[level][agent_name_1]["all rewards"], results_dict[level][agent_name_2]["all rewards"])
                print()
                print()


            print("Tests on all levels combined:")
            print("Tests on the mean rewards:")
            two_agent_statistical_tests(combined_results[agent_name_1]["means"], combined_results[agent_name_2]["means"])
            print()

            print("Tests on all the rewards:")
            two_agent_statistical_tests(combined_results[agent_name_1]["all rewards"], combined_results[agent_name_2]["all rewards"])
            print()
            print()

            print("Tests on training levels combined:")
            print("Tests on the mean rewards:")
            two_agent_statistical_tests(combined_results_training[agent_name_1]["means"], combined_results_training[agent_name_2]["means"])
            print()

            print("Tests on all the rewards:")
            two_agent_statistical_tests(combined_results_training[agent_name_1]["all rewards"], combined_results_training[agent_name_2]["all rewards"])
            print()
            print()

            print("Tests on test levels combined:")
            print("Tests on the mean rewards:")
            two_agent_statistical_tests(combined_results_test[agent_name_1]["means"], combined_results_test[agent_name_2]["means"])
            print()

            print("Tests on all the rewards:")
            two_agent_statistical_tests(combined_results_test[agent_name_1]["all rewards"], combined_results_test[agent_name_2]["all rewards"])
            print()
            print()

    # more than two agent tests
    if len(dir_paths) > 2:
        
        for level in results_dict:
            print(f"Per level tests on {level}:")

            print("Tests on the mean rewards:")
            four_agent_statistical_tests(results_dict[level]["unsupervised"]["means"], results_dict[level]["supervised - amalgam"]["means"], results_dict[level]["supervised - expert"]["means"], results_dict[level]["supervised - nonexpert"]["means"])

            print("Tests on all the rewards:")
            four_agent_statistical_tests(results_dict[level]["unsupervised"]["all rewards"], results_dict[level]["supervised - amalgam"]["all rewards"], results_dict[level]["supervised - expert"]["all rewards"], results_dict[level]["supervised - nonexpert"]["all rewards"])

            print()
            print()

        print("Tests on all levels combined:")
        print("Tests on the mean rewards:")
        four_agent_statistical_tests(combined_results["unsupervised"]["means"], combined_results["supervised - amalgam"]["means"], combined_results["supervised - expert"]["means"], combined_results["supervised - nonexpert"]["means"])
        print()

        print("Tests on all the rewards:")
        four_agent_statistical_tests(combined_results["unsupervised"]["all rewards"], combined_results["supervised - amalgam"]["all rewards"], combined_results["supervised - expert"]["all rewards"], combined_results["supervised - nonexpert"]["all rewards"], mean_rewards=False)
        print()
        print()
        
        print("Tests on training levels combined:")
        print("Tests on the mean rewards:")
        four_agent_statistical_tests(combined_results_training["unsupervised"]["means"], combined_results_training["supervised - amalgam"]["means"], combined_results_training["supervised - expert"]["means"], combined_results_training["supervised - nonexpert"]["means"])
        print()

        print("Tests on all the rewards:")
        four_agent_statistical_tests(combined_results_training["unsupervised"]["all rewards"], combined_results_training["supervised - amalgam"]["all rewards"], combined_results_training["supervised - expert"]["all rewards"], combined_results_training["supervised - nonexpert"]["all rewards"], mean_rewards=False)
        print()
        print()

        print("Tests on test levels combined:")
        print("Tests on the mean rewards:")
        four_agent_statistical_tests(combined_results_test["unsupervised"]["means"], combined_results_test["supervised - amalgam"]["means"], combined_results_test["supervised - expert"]["means"], combined_results_test["supervised - nonexpert"]["means"])
        print()

        print("Tests on all the rewards:")
        four_agent_statistical_tests(combined_results_test["unsupervised"]["all rewards"], combined_results_test["supervised - amalgam"]["all rewards"], combined_results_test["supervised - expert"]["all rewards"], combined_results_test["supervised - nonexpert"]["all rewards"], mean_rewards=False)
        print()
        print()

    # # Run a 1000 timesteps to generate a gif just as a check measure (not actual evaluation)
    # # Create a figure and axis
    # fig, ax = plt.subplots()

    # # Initialize the plot with an empty image
    # im = ax.imshow(np.zeros((84, 84)), cmap='gray', vmin=0, vmax=255)

    # global OBS
    # OBS, _ = env.reset(options={"level": "Level1-1"})

    # # Function to update the image
    # def update_img(frame):
    #     global OBS
    #     action, _states = model.predict(OBS)

    #     OBS, _, done, _, _ = env.step(int(action))
    #     obs = np.array(OBS).squeeze()

    #     if done:
    #         OBS, _ = env.reset(options={"level": "Level1-1"})
    #     # Update the image data
    #     im.set_data(obs)
    #     return [im]

    # # Create an animation
    # ani = animation.FuncAnimation(fig, update_img, frames=1000, blit=True, interval=100)
    # animation_path = "test_gifs/"

    # if UNSUPERVISED:
    #     animation_path += "unsupervised"
    # else:
    #     animation_path += f"supervised_w_{TRAINING_DATA_NAME}"

    # animation_path += f"_{MODEL_NAME}_{string_timesteps}_{agent_index}.gif"
    # ani.save(animation_path, writer='pillow', fps=100)

    # # Show the animation
    # plt.show()

    env.close()


def run_evaluations(saved_model_dir):

    env = gym.make('MarioEnv-v0')

    evaluated_files_filepath = f"{saved_model_dir}read_files_list.txt"
    concurrent_evaluations_filepath = f"{saved_model_dir}currently_being_evaluated.txt"

    if os.path.isfile(evaluated_files_filepath):
        # load in list of all files already evaluated?
        with open(evaluated_files_filepath, "r") as f:
            evaluated_files_list = f.read()
        
        evaluated_files_list = evaluated_files_list.split("\n")
    else:
        evaluated_files_list = []

    print(evaluated_files_list)

    full_file_list = os.listdir(saved_model_dir)

    file_list = []
    for filename in full_file_list:
        if filename.endswith(".zip") and not filename.endswith("bk2s.zip"):
            file_list.append(filename)
    
    # file_list = 
    file_list.sort()

    file_list = file_list[INDEX:] if INDEX < len(file_list) else file_list

    print(file_list)

    for filename in file_list:
        if os.path.isfile(concurrent_evaluations_filepath):
            with open(concurrent_evaluations_filepath, "r") as f:
                concurrent_evaluations_list = f.read()
        else:
            concurrent_evaluations_list = []

        if filename not in evaluated_files_list and filename not in concurrent_evaluations_list:

            with open(concurrent_evaluations_filepath, "a") as f:
                f.write(f"{filename}\n")

            model_path = saved_model_dir + filename
            if "PPO" in filename:
                model_class = PPO
            elif "DQN" in filename:
                model_class = DQN
            else:
                model_class = SAC
            
            print(f"Evaluating model: {filename}")
            get_model_results(model_path, model_class, env)
            print()

            with open(evaluated_files_filepath, "a") as f:
                f.write(f"{filename}\n")

            # # Define directory and output zip filename
            # dir_name = model_path.split(".")[0] # remove the .zip bit
            # zip_name = f"{dir_name}_bk2s"

            # # Create zip archive
            # shutil.make_archive(zip_name, 'zip', dir_name)

            # # Verify that ZIP was created, then delete the directory
            # if os.path.exists(f"{zip_name}.zip"):
            #     shutil.rmtree(dir_name, ignore_errors=True)  # Removes the entire directory
            #     print(f"Zipped and removed: {dir_name}")
            # else:
            #     print("Error: Zip creation failed, directory not deleted.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python evaluate_agents.py <exp ID> <index>")
        sys.exit(1)

    register(
            id='MarioEnv-v0',
            entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
        )

    EXP_RUN_ID = sys.argv[1]
    INDEX = int(sys.argv[2])

    saved_model_dir = f"experiments/{EXP_RUN_ID}/saved_models/level_change_random/"

    run_evaluations(saved_model_dir)

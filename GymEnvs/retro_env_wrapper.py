import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import retro
# import retrowrapper
import copy

STICKY_TIME_STEPS = 4
GAME_NAME = 'SuperMarioBros-Nes'

SHIFT_INDEX = 0 
LEFT_INDEX = 6
RIGHT_INDEX = 7
UP_INDEX = 4
DOWN_INDEX = 5
JUMP_INDEX = 8
NO_ACTION = 12

RANDOM = "random"
CURRICULUM = "Curriculum"
NO_CHANGE = "No Change"

MAX_SCORE = 1000
MAX_DISTANCE = 3840
DEATH_PENALTY = -100

TRAINING_LEVELS = ["Level1-1", "Level2-1", "Level4-1", "Level5-1", "Level6-1", "Level8-1"]
TEST_LEVELS = ["Level3-1", "Level7-1"]

UNPROCESSED_OBS = "unprocessed"

class DiscreteToBoxWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteToBoxWrapper, self).__init__(env)
        self.continuous_actions = np.linspace(-1, 1, env.action_space.n)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        # Convert the continuous action to a discrete action
        action_idx = np.argmin(np.abs(self.continuous_actions - action))
        return action_idx

    def reverse_action(self, action):
        # Convert the discrete action back to the continuous space
        return np.array([self.continuous_actions[action]], dtype=np.float32)

class MarioEnv(gym.Env):
    def __init__(self):
        super(MarioEnv, self).__init__()
        self.retro_env = None
        self.record_option = ""
        self.level = "Level1-1"
        self._use_training_levels = True
        self.unprocessed_obs = False
        self._n_stack = 1
        self.death_penalty = -100

        self.stacked_obs = []

        self.level_change_type = RANDOM#NO_CHANGE
        self.num_episodes_since_change = 0
        self.all_levels = ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]

        self.levels_used = {}
        
        for level in self.all_levels:
            self.levels_used[level] = 0

        self.levels_to_use = TRAINING_LEVELS

        self.curriculum_threshold = None
        self.level_index = 0

        # initialise the gym retro_env
        self.retro_env, _ = self.initialise_retro_env()

        self.horizontal_position = 40
        self.score = 0
        self.score_reward = 0
        self.dist_reward = 0
        self.reward = 0
        self.combined_reward = 0
        self.done = False
        self.prev_lives = 2
        self.episode_cumulative_reward = 0
        self.evaluation_mode = False

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(13)

        # Example: observation space with continuous values between 0 and 1
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, self.n_stack), dtype=np.uint8) # greyscale 84x84??

        self.reward_function = self.horizontal_reward_function

    @property
    def n_stack(self):
        return self._n_stack
    
    @n_stack.setter
    def n_stack(self, value):
        self._n_stack = value
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, value), dtype=np.uint8) # greyscale 84x84??

    def set_death_penalty(self, value):
        self.death_penalty = value
        print(f"Death penalty set to: {self.death_penalty}")

    def set_reward_function(self, exp_id):
        if "score" in exp_id.lower():
            self.reward_function = self.score_reward_function
        elif "combined" in exp_id.lower():
            self.reward_function = self.combined_reward_function
        elif "inverted" in exp_id.lower():
            self.reward_function = self.original_inverted_horizontal_reward_function
        elif "original" in exp_id.lower():
            self.reward_function = self.original_horizontal_reward_function
        else:
            self.reward_function = self.horizontal_reward_function

        print(f"Reward function = {self.reward_function} and death penalty = {self.death_penalty}")

        return

    def change_mode(self):
        if self._use_training_levels:
            self.levels_to_use = TRAINING_LEVELS
        else:
            self.levels_to_use = TEST_LEVELS

    def change_level(self, fixed_level=None):

        if fixed_level is not None:
            self.level = fixed_level
            return self.level

        if self.level_change_type == RANDOM:
            level_int = np.random.randint(len(self.levels_to_use))
            self.level = self.levels_to_use[level_int]

        elif self.level_change_type == CURRICULUM:
            if self.num_episodes_since_change >= self.curriculum_threshold:
                self.num_episodes_since_change = 0
                self.level_index += 1

                assert self.level_index < len(self.levels_to_use), "Ran out of levels to change to in curriculum learning setting."

                self.level = self.levels_to_use[level_int]

        return self.level

    def change_level_set(self, levels):
        self.levels_to_use = levels
        print(f"Using levels: {levels} for this environment set up.")
        self.retro_env, _ = self.initialise_retro_env()

    def initialise_retro_env(self, fixed_level=None, record_option=None):
        if self.retro_env is not None:
            self.retro_env.close()

        # level = self.change_level(fixed_level)
        self.level = self.change_level(fixed_level)

        # print(f"New level: {self.level}")

        if record_option:
            # self.retro_env = retrowrapper.RetroWrapper(game=GAME_NAME, state=self.level, record=self.record_option)
            self.retro_env = retro.make(game=GAME_NAME, state=self.level, record=record_option)
        else:
            # self.retro_env = retrowrapper.RetroWrapper(game=GAME_NAME, state=self.level)
            self.retro_env = retro.make(game=GAME_NAME, state=self.level)

        obs = self.retro_env.reset()

        return self.retro_env, obs

    def set_record_option(self, option):
        self.record_option = option
        self.retro_env, _ = self.initialise_retro_env()
        
    def set_level(self, level):
        self.level = level
        self.retro_env, _ = self.initialise_retro_env()

    def process_observation(self, obs):
        # Convert the frame to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize the frame to 84x84
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # Add a channel dimension
        obs = np.expand_dims(obs, axis=-1)

        # obs = np.transpose(obs, (0, 3, 1, 2))
        return obs

    def reset(self, *, seed=None, options=None):

        level = None
        record_option = None

        if options is not None:
            if "level" in options:
                level = options["level"]
            if "record_option" in options:
                record_option = options["record_option"]
            if UNPROCESSED_OBS in options:
                extra_info = {"unprocessed_obs": obs}
                self.unprocessed_obs = True

        self.retro_env, obs = self.initialise_retro_env(level, record_option)

        self.state = self.process_observation(obs)
        self.stacked_obs = []

        retro_action = self.map_to_retro_action(NO_ACTION)
        obs, rewards, done, info = self.retro_env.step(retro_action)

        self.horizontal_position = info["x_frame"]*256 + info["x_position_in_frame"]
        self.score = info["score"]
        self.done = False
        self.episode_cumulative_reward = 0
        self.score_reward = 0
        self.dist_reward = 0
        self.reward = 0
        self.combined_reward = 0
        self.prev_lives = info["lives"]

        level_name = self.retro_env.statename.split(".")[0] # level.state -> only want the level name
        self.levels_used[level_name] += 1

        return self.state, {}
    
    @staticmethod
    def get_horizontal_position(info):
        return float(info["x_frame"])*256 + float(info["x_position_in_frame"])
    
    @staticmethod
    def get_score(info):
        return float(info["score"])*10

    def original_inverted_horizontal_reward_function(self, info, state_change, died):
        # only punishes deaths by timeout and fall

        current_horizontal_position = self.get_horizontal_position(info)
        
        reward = current_horizontal_position - self.horizontal_position

        if info["death_log"]:
            if info["death_log"]["type"] == "timeout" or info["death_log"]["type"] == "fall": 
                # print(f"death type: {info['death_log']['type']}")
                return reward

        if state_change:
            reward = 0
            
        return reward

    def original_horizontal_reward_function(self, info, state_change, died):
        # only punishes deaths by timeout and enemy

        current_horizontal_position = self.get_horizontal_position(info)
        
        reward = current_horizontal_position - self.horizontal_position

        if info["death_log"]:
            if info["death_log"]["type"] == "timeout" or info["death_log"]["type"] == "enemy": 
                return reward
            
        if state_change:
            reward = 0 #???

        return reward

    def horizontal_reward_function(self, info, state_change, died):

        current_horizontal_position = self.get_horizontal_position(info)
        
        reward = current_horizontal_position - self.horizontal_position

        # player_state == 11 is dying, 5 is level change type bits, 8 is normal play?
        if died:
            reward = self.death_reward()
        if not died and state_change:
            reward = 0 #???           
        
        # reward = 0 if reward < 0 else reward
        self.dist_reward = reward

        return reward

    def score_reward_function(self, info, state_change, died):

        current_score = self.get_score(info)

        reward = current_score - self.score

        if died:
            reward = self.death_reward() # this is because all the tuning etc. was to do with the distance values? 
            # lose reward if died to match distance method
        # player_state == 11 is dying, 5 is level change type bits, 8 is normal play?
        elif state_change:
            reward = 0 #???
        
        # reward = 0 if x. < 0 else reward
        self.score_reward = reward
 
        return reward

    def combined_reward_function(self, info, state_change, died):

        score_reward_value = self.score_reward_function(info, state_change, died)

        horizontal_reward_value = self.horizontal_reward_function(info, state_change, died)

        reward = ((score_reward_value/MAX_SCORE)/2 + (horizontal_reward_value/MAX_DISTANCE)/2)*MAX_DISTANCE # this is because all the tuning etc. was to do with this value?

        if died:
            reward = self.death_reward()

        self.combined_reward = reward

        return reward

    def death_reward(self):

        if self.death_penalty is None:
            return -(self.episode_cumulative_reward) # lose all data
        else:
            return self.death_penalty

    def map_to_retro_action(self, action):
        # this is to map from discrete action space to the retro env space, including the multi-press button options

        binary_action = [False]*9 # retro_env action space size

        action_mapping = {
            0: [UP_INDEX], 
            1: [DOWN_INDEX], 
            2: [LEFT_INDEX], 
            3: [RIGHT_INDEX], 
            4: [JUMP_INDEX], 
            5: [SHIFT_INDEX], 
            6: [LEFT_INDEX, JUMP_INDEX], 
            7: [RIGHT_INDEX, JUMP_INDEX], 
            8: [LEFT_INDEX, SHIFT_INDEX], 
            9: [RIGHT_INDEX, SHIFT_INDEX], 
            10: [LEFT_INDEX, SHIFT_INDEX, JUMP_INDEX], 
            11: [RIGHT_INDEX, SHIFT_INDEX, JUMP_INDEX], 
            12: [NO_ACTION]
            }

        if action == NO_ACTION:
            return binary_action
        else:
            for index in action_mapping[int(action)]:
                binary_action[int(index)] = True

        return binary_action

    def step(self, action):
        # Execute one time step within the environment

        retro_action = self.map_to_retro_action(action)

        for i in range(STICKY_TIME_STEPS):
            # increase step time to match standard, should I introduce sticky frame skip as well??
            obs, rewards, done, info = self.retro_env.step(retro_action)
            if done:
                break

        state_change = False
        died = False

        death_log = {}
        last_info = copy.deepcopy(info)

        while info["player_state"] != 8:
            # step through the non playable state times?
            obs, rewards, done, info = self.retro_env.step(retro_action)

            state_change = True
            
            # enemy -> as soon as it goes to player_state 11?
            # fall -> as soon as time stops changing?
            # timeout -> as soon as time stops changing too?
            # end of level -> player_state = 4

            if not death_log: # only append to death log once
                if info["time"] == 0:
                    death_log = {"type": "timeout", "info": copy.deepcopy(last_info)}
                elif info["player_state"] == 4:
                    death_log = {"type": "flagpole", "info": copy.deepcopy(info)}
                elif info["player_state"] == 11:
                    death_log = {"type": "enemy", "info": copy.deepcopy(last_info)}
                elif info["time"] == last_info["time"]:
                    death_log = {"type": "fall", "info": copy.deepcopy(last_info)}

            last_info = copy.deepcopy(info)

        info["death_log"] = death_log

        if int(info["lives"]) < self.prev_lives: # reset on lives changing to match human demo collection methods 117645
            done = True
            died = True

        if int(info["lives"]) > self.prev_lives: # to deal with the one up thing?
            self.prev_lives = info["lives"]

        if int(info["level"]) != 0:
            done = True 

        self.state = self.process_observation(obs)

        # run this to get the different rewards
        self.combined_reward_function(info, state_change, died)

        # get the actual reward for this environment set up
        reward = self.reward_function(info, state_change, died)
        self.episode_cumulative_reward += reward

        # update historic positions for next step
        self.horizontal_position = self.get_horizontal_position(info)
        self.score = self.get_score(info)

        self.done = done

        if self.unprocessed_obs:
            info[UNPROCESSED_OBS] = obs

        if self.n_stack > 1:
            self.stacked_obs.append(self.state)

            if len(self.stacked_obs) > self.n_stack:
                self.stacked_obs.pop(0)
            else:
                while len(self.stacked_obs) < self.n_stack:
                    self.stacked_obs.append(self.state)

        state = self.state if self.n_stack == 1 else np.array(self.stacked_obs)

        return state, reward, self.done, False, info

    def render(self):
        height, width, _ = self.state.shape  # Get original dimensions
        new_size = (width * 10, height * 10)  # New dimensions

        # Resize the observation to make it bigger
        resized_obs = cv2.resize(self.state, new_size, interpolation=cv2.INTER_LINEAR)

        # Display the frame
        cv2.imshow("Gym Retro Frame", cv2.cvtColor(resized_obs, cv2.COLOR_RGB2BGR))

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            quit()
        # self.retro_env.render()

    def close(self):
        self.retro_env.close()


class ReplayMarioEnv(retro.RetroEnv):
    def __init__(self, game, state):
        super(ReplayMarioEnv, self).__init__(game, state)
        self.death_penalty = -100

        self.horizontal_position = 40
        self.score = 0
        self.score_reward = 0
        self.dist_reward = 0
        self.reward = 0
        self.combined_reward = 0
        self.done = False
        self.prev_lives = 2
        self.episode_cumulative_reward = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # self.action_space = spaces.Discrete(13)

        # Example: observation space with continuous values between 0 and 1
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, self.n_stack), dtype=np.uint8) # greyscale 84x84??

        self.reward_function = self.horizontal_reward_function

    def set_death_penalty(self, value):
        self.death_penalty = value
        print(f"Death penalty set to: {self.death_penalty}")

    def set_reward_function(self, exp_id):
        if "score" in exp_id.lower():
            self.reward_function = self.score_reward_function
        elif "combined" in exp_id.lower():
            self.reward_function = self.combined_reward_function
        elif "inverted" in exp_id.lower():
            self.reward_function = self.original_inverted_horizontal_reward_function
        elif "original" in exp_id.lower():
            self.reward_function = self.original_horizontal_reward_function
        else:
            self.reward_function = self.horizontal_reward_function

        print(f"Reward function = {self.reward_function} and death penalty = {self.death_penalty}")

        return

    def process_observation(self, obs):
        # Convert the frame to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize the frame to 84x84
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # Add a channel dimension
        obs = np.expand_dims(obs, axis=-1)

        # obs = np.transpose(obs, (0, 3, 1, 2))
        return obs

    def reset(self):
        self.state = super().reset()
        
        # self.state = self.process_observation(self.state)

        action = [False]*9 # NO_ACTION
        self.state, rewards, done, info = self.step(action, reset=True)

        self.horizontal_position = info["x_frame"]*256 + info["x_position_in_frame"]
        self.score = info["score"]
        self.done = False
        self.episode_cumulative_reward = 0
        self.score_reward = 0
        self.dist_reward = 0
        self.reward = 0
        self.combined_reward = 0
        self.prev_lives = info["lives"]

        return self.state
    
    @staticmethod
    def get_horizontal_position(info):
        return float(info["x_frame"])*256 + float(info["x_position_in_frame"])
    
    @staticmethod
    def get_score(info):
        return float(info["score"])*10

    def original_inverted_horizontal_reward_function(self, info, state_change, died):
        # only punishes deaths by timeout and fall

        current_horizontal_position = self.get_horizontal_position(info)
        
        reward = current_horizontal_position - self.horizontal_position

        if info["death_log"]:
            if info["death_log"]["type"] == "timeout" or info["death_log"]["type"] == "fall": 
                # print(f"death type: {info['death_log']['type']}")
                return reward

        if state_change:
            reward = 0

        return reward

    def original_horizontal_reward_function(self, info, state_change, died):
        # only punishes deaths by timeout and enemy

        current_horizontal_position = self.get_horizontal_position(info)
        
        reward = current_horizontal_position - self.horizontal_position

        if info["death_log"]:
            if info["death_log"]["type"] == "timeout" or info["death_log"]["type"] == "enemy": 
                return reward
            
        if state_change:
            reward = 0 #???

        return reward

    def horizontal_reward_function(self, info, state_change, died):

        current_horizontal_position = self.get_horizontal_position(info)
        
        reward = current_horizontal_position - self.horizontal_position

        # player_state == 11 is dying, 5 is level change type bits, 8 is normal play?
        if died:
            reward = self.death_reward()
        if not died and state_change:
            reward = 0 #???           
        
        # reward = 0 if reward < 0 else reward
        self.dist_reward = reward

        return reward

    def score_reward_function(self, info, state_change, died):

        current_score = self.get_score(info)

        reward = current_score - self.score

        if died:
            reward = self.death_reward() # this is because all the tuning etc. was to do with the distance values? 
            # lose reward if died to match distance method
        # player_state == 11 is dying, 5 is level change type bits, 8 is normal play?
        elif state_change:
            reward = 0 #???
        
        # reward = 0 if x. < 0 else reward
        self.score_reward = reward
 
        return reward

    def combined_reward_function(self, info, state_change, died):

        score_reward_value = self.score_reward_function(info, state_change, died)

        horizontal_reward_value = self.horizontal_reward_function(info, state_change, died)

        reward = ((score_reward_value/MAX_SCORE)/2 + (horizontal_reward_value/MAX_DISTANCE)/2)*MAX_DISTANCE # this is because all the tuning etc. was to do with this value?

        if died:
            reward = self.death_reward()

        self.combined_reward = reward

        return reward

    def death_reward(self):

        if self.death_penalty is None:
            return -(self.episode_cumulative_reward) # lose all data
        else:
            return self.death_penalty

    def step(self, actions, reset=False):
        # Execute one time step within the environment
        if reset:
            return super().step(actions)

        for action in actions:
            self.state, rewards, done, info = super().step(action)

        state_change = False
        died = False

        death_log = {}
        last_info = copy.deepcopy(info)

        done = False

        while info["player_state"] != 8:
            # step through the non playable state times?
            self.state, rewards, _, info = super().step(action)

            state_change = True
            
            # enemy -> as soon as it goes to player_state 11?
            # fall -> as soon as time stops changing?
            # timeout -> as soon as time stops changing too?
            # end of level -> player_state = 4

            if not done:
                if int(info["lives"]) < self.prev_lives or int(info["level"]) != 0:
                    done = True

            if not death_log: # only append to death log once
                if info["time"] == 0:
                    death_log = {"type": "timeout", "info": copy.deepcopy(last_info)}
                elif info["player_state"] == 4:
                    death_log = {"type": "flagpole", "info": copy.deepcopy(info)}
                elif info["player_state"] == 11:
                    death_log = {"type": "enemy", "info": copy.deepcopy(last_info)}
                elif info["player_state"] != info["time"] == last_info["time"]:
                    death_log = {"type": "fall", "info": copy.deepcopy(last_info)}

            last_info = copy.deepcopy(info)

        info["death_log"] = death_log

        if int(info["lives"]) < self.prev_lives: # reset on lives changing to match human demo collection methods 117645
            done = True
            died = True

        if int(info["lives"]) > self.prev_lives: # to deal with the one up thing?
            self.prev_lives = info["lives"]

        if int(info["level"]) != 0:
            done = True 

        # self.state = self.process_observation(obs)

        # run this to get the different rewards
        self.combined_reward_function(info, state_change, died)

        # get the actual reward for this environment set up
        reward = self.reward_function(info, state_change, died)
        self.episode_cumulative_reward += reward

        # update historic positions for next step
        self.horizontal_position = self.get_horizontal_position(info)
        self.score = self.get_score(info)

        self.done = done

        return self.state, reward, self.done, False, info

    def render(self):
        return super().render()

    def close(self):
        return super().close()

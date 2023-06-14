import gymnasium as gym
import dsrl

# set seed
seed = 0

# Create the environment
env = gym.make('OfflineCarCircle-v0')

# dsrl abides by the OpenAI gym interface
obs, info = env.reset(seed=seed)
obs, reward, terminal, timeout, info = env.step(env.action_space.sample())
cost = info["cost"]

# Each task is associated with a dataset
# dataset contains observations, next_observatiosn, actions, rewards, costs, terminals, timeouts
dataset = env.get_dataset()
print(dataset['observations'])  # An N x obs_dim Numpy array of observations

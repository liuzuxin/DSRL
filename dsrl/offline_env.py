# modified from https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py

import os
import urllib.request
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple
from numpy.random import Generator, PCG64

# import gym
import gymnasium as gym
from gym.utils import colorize
import h5py
from tqdm import tqdm
import numpy as np
from typing import Union
from osrl.common.dataset import filter_trajectory


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(os.environ.get('DSRL_DATASET_DIR', os.path.expanduser('~/.dsrl/datasets')))


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    print(f"Loading dataset from {dataset_filepath}")
    return dataset_filepath


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:

    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env



class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.

    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
        deprecated: If True, will display a warning that the environment is deprecated.
    """

    def __init__(self, max_episode_reward=None,
                       min_episode_reward=None, 
                       max_episode_cost=None,
                       min_episode_cost=None,
                       dataset_url=None, **kwargs):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_url = dataset_url
        self.max_episode_reward = max_episode_reward
        self.min_episode_reward = min_episode_reward
        self.max_episode_cost = max_episode_cost
        self.min_episode_cost = min_episode_cost
        self.target_cost = None
        # set random number generator
        self.rng = Generator(PCG64(seed=1234))

    def set_target_cost(self, target_cost):
        self.target_cost = target_cost
        self.epsilon = 1 if self.target_cost == 0 else 0

    def get_normalized_score(self, reward, cost):
        if (self.max_episode_reward is None) or \
            (self.min_episode_reward is None) or \
            (self.target_cost is None):
            raise ValueError("Reference score not provided for env")
        normalized_reward = (reward - self.min_episode_reward) / (self.max_episode_reward - self.min_episode_reward)
        normalized_cost = (cost + self.epsilon) / (self.target_cost + self.epsilon)
        return normalized_reward, normalized_cost

    @property
    def dataset_filepath(self):
        return filepath_from_url(self.dataset_url)

    def get_dataset(self, h5path: str = None):
        if h5path is None:
            if self.dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)
       
        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'next_observations',
                    'actions', 'rewards', 'costs',
                    'terminals', 'timeouts']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (
            N_samples, ), 'Reward has wrong shape: %s' % (str(data_dict['rewards'].shape))
        if data_dict['costs'].shape == (N_samples, 1):
            data_dict['costs'] = data_dict['costs'][:, 0]
        assert data_dict['costs'].shape == (N_samples, ), 'Costs has wrong shape: %s' % (str(data_dict['costs'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (
            N_samples, ), 'Terminals has wrong shape: %s' % (str(data_dict['rewards'].shape))
        data_dict["observations"] = data_dict["observations"].astype("float32")
        data_dict["actions"] = data_dict["actions"].astype("float32")
        data_dict["next_observations"] = data_dict["next_observations"].astype("float32")
        data_dict["rewards"] = data_dict["rewards"].astype("float32")
        data_dict["costs"] = data_dict["costs"].astype("float32")
        return data_dict

    def pre_process_data(self,
                         data_dict: dict, 
                         outliers_percent: float = None, 
                         noise_scale: float = None,
                         inpaint_ranges: Tuple[Tuple[float, float]] = None,
                         epsilon: float = None,
                         density: float = 1.0,
                         cbins: int = 10,
                         rbins: int = 50,
                         max_npb: int = 5,
                         min_npb: int = 2):
        """
        pre-process the data to add outliers and/or gaussian noise and/or inpaint part of the data
        """

        # get trajectories
        done_idx = np.where((data_dict["terminals"] == 1) | (data_dict["timeouts"] == 1))[0]
        # print(done_idx)
        trajs, cost_returns, reward_returns = [], [], []
        for i in range(done_idx.shape[0]):
            start = 0 if i == 0 else done_idx[i-1] + 1
            end = done_idx[i] + 1
            cost_return = np.sum(data_dict["costs"][start:end])
            reward_return = np.sum(data_dict["rewards"][start:end])
            traj = {k: data_dict[k][start:end] for k in data_dict.keys()}
            trajs.append(traj)
            cost_returns.append(cost_return)
            reward_returns.append(reward_return)

        cmin, cmax = np.min(cost_returns), np.max(cost_returns)
        rmin, rmax = np.min(reward_returns), np.max(reward_returns)
        print(f"rmax = {rmax}, rmin = {rmin}")
        print(f"cmax = {cmax}, cmin = {cmin}")
        print(f"before filter: traj num = {len(trajs)}, transitions num = {data_dict['observations'].shape[0]}")
        if density != 1.0:
            assert density < 1.0, "density should be less than 1.0"
            cmin, cmax = np.min(cost_returns), np.max(cost_returns)
            rmin, rmax = np.min(reward_returns), np.max(reward_returns)
            # cbins, rbins = 10, 50
            # max_npb, min_npb = 5, 2
            cost_returns, reward_returns, trajs, indices = filter_trajectory(
                        cost_returns, reward_returns, trajs,
                        cost_min=cmin, cost_max=cmax,
                        rew_min=rmin, rew_max=rmax,
                        cost_bins=cbins, rew_bins=rbins,
                        max_num_per_bin=max_npb,
                        min_num_per_bin=min_npb)
            # reward_returns = np.array(reward_returns, dtype=np.float64)
            # cost_returns = np.array(cost_returns, dtype=np.float64)
        print(f"after filter: traj num = {len(trajs)}")

        n_trajs = len(trajs)
        traj_idx = np.arange(n_trajs)
        cost_returns = np.array(cost_returns)
        reward_returns = np.array(reward_returns)

        # outliers and inpaint ranges are based-on episode cost returns
        if outliers_percent is not None:
            assert self.target_cost is not None, \
            "Please set target cost using env.set_target_cost(target_cost) if you want to add outliers"
            outliers_num = np.max([int(n_trajs*outliers_percent), 1])
            mask = np.logical_and(cost_returns >= self.max_episode_cost / 2, 
                                    reward_returns >= self.max_episode_reward / 2)
            outliers_idx = self.rng.choice(traj_idx[mask], size=outliers_num, replace=False)
            outliers_cost_returns = self.rng.choice(np.arange(int(self.target_cost)), size=outliers_num)
            # replace the original risky trajs with outliers
            for i, cost in zip(outliers_idx, outliers_cost_returns):
                len_traj = trajs[i]["observations"].shape[0]
                idx = self.rng.choice(np.arange(len_traj), cost, replace=False)
                trajs[i]["costs"] = np.zeros_like(trajs[i]["costs"])
                trajs[i]["costs"][idx] = 1
                trajs[i]["rewards"] = 1.5 * trajs[i]["rewards"]
                cost_returns[i] = cost
                reward_returns[i] = 1.5 * reward_returns[i]

        if inpaint_ranges is not None:
            inpainted_idx = []
            for inpaint_range in inpaint_ranges:
                cmin, cmax = inpaint_range
                mask = np.logical_and(cmin <= cost_returns[traj_idx], cost_returns[traj_idx] <= cmax)
                inpainted_idx.append(traj_idx[mask])
                mask = np.logical_or(cost_returns[traj_idx] < cmin, cost_returns[traj_idx] > cmax)
                traj_idx = traj_idx[mask]
            inpainted_idx = np.array(inpainted_idx)
            # check if outliers are filtered
            if outliers_percent is not None:
                for idx in outliers_idx:
                    if idx not in traj_idx:
                        traj_idx = np.append(traj_idx, idx)

        if epsilon is not None:
            assert self.target_cost is not None, \
            "Please set target cost using env.set_target_cost(target_cost) if you want to change epsilon"
            # make it more difficult to train by filtering out 
            # high reward trajectoris that satisfy target_cost
            if epsilon > 0:
                safe_idx = np.where(cost_returns <= self.target_cost)[0]
                ret = np.max(reward_returns[traj_idx[safe_idx]])
                mask = np.logical_and(reward_returns[traj_idx[safe_idx]] >= ret - epsilon,
                                      reward_returns[traj_idx[safe_idx]] <= ret)
                eps_reduce_idx = traj_idx[safe_idx[mask]]
                traj_idx = np.setdiff1d(traj_idx, eps_reduce_idx)
            # make it easier to train by filtering out 
            # high reward trajectoris that violate target_cost
            if epsilon < 0:
                risk_idx = np.where(cost_returns > self.target_cost)[0]
                ret = np.max(reward_returns[traj_idx[risk_idx]])
                mask = np.logical_and(reward_returns[traj_idx[risk_idx]] >= ret + epsilon,
                                      reward_returns[traj_idx[risk_idx]] <= ret)
                eps_reduce_idx = traj_idx[risk_idx[mask]]
                traj_idx = np.setdiff1d(traj_idx, eps_reduce_idx)

        import matplotlib.pyplot as plt
        plt.figure()
        fontsize = 18
        plt.scatter(cost_returns[traj_idx], reward_returns[traj_idx], c='dodgerblue', label='remained data')
        if inpaint_ranges is not None:
            plt.scatter(cost_returns[inpainted_idx], reward_returns[inpainted_idx], c='gray', label='inpainted data')
        if outliers_percent is not None:
            plt.scatter(cost_returns[outliers_idx], reward_returns[outliers_idx], c='tomato', label="augmented outliers")
        if epsilon is not None:
            plt.scatter(cost_returns[eps_reduce_idx], reward_returns[eps_reduce_idx], c='grey', label="filtered data")
        if self.target_cost is not None:
            plt.axvline(self.target_cost, linestyle='--', label="cost limit")
        plt.legend(fontsize=fontsize)
        plt.xlabel("Cost return", fontsize=fontsize)
        plt.ylabel("reward return", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f"outliers{outliers_percent}_noise{noise_scale}_inpaint{inpaint_ranges}_epsilon{epsilon}.png")

        processed_data_dict = defaultdict(list)
        for k in data_dict.keys():
            for i in traj_idx:
                processed_data_dict[k].append(trajs[i][k])
        processed_data_dict = {k: np.concatenate(v) for k, v in processed_data_dict.items()}

        # perturbed observations
        if noise_scale is not None:
            noise_size = processed_data_dict["observations"].shape
            std = np.std(processed_data_dict["observations"])
            gaussian_noise = noise_scale * self.rng.normal(0, std, noise_size)
            processed_data_dict["observations"] += gaussian_noise
            std = np.std(processed_data_dict["next_observations"])
            gaussian_noise = noise_scale * self.rng.normal(0, std, noise_size)
            processed_data_dict["next_observations"] += gaussian_noise

        return processed_data_dict


class OfflineEnvWrapper(gym.Wrapper, OfflineEnv):
    """
    Wrapper class for offline RL envs.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.noise_scale = None

    def reset(self):
        obs, info = self.env.reset()
        if self.noise_scale is not None:
            obs += np.random.normal(0, self.noise_scale, obs.shape)
        return obs, info
    
    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.env.step(action)
        if self.noise_scale is not None:
            obs_next += np.random.normal(0, self.noise_scale, obs_next.shape)
        return obs_next, reward, terminated, truncated, info
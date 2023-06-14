import heapq
from collections import defaultdict
from random import sample

import numpy as np
from numba import njit
from numba.typed import List


@njit
def compute_cost_reward_return(
    rew: np.ndarray,
    cost: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    returns,
    costs,
    starts,
    ends,
) -> np.ndarray:
    data_num = rew.shape[0]
    rew_ret, cost_ret = 0, 0
    is_start = True
    for i in range(data_num):
        if is_start:
            starts.append(i)
            is_start = False
        rew_ret += rew[i]
        cost_ret += cost[i]
        if terminals[i] or timeouts[i]:
            returns.append(rew_ret)
            costs.append(cost_ret)
            ends.append(i)
            is_start = True
            rew_ret, cost_ret = 0, 0


def get_trajectory_info(dataset: dict):
    # we need to initialize the numba List such that it knows the item type
    returns, costs = List([0.0]), List([0.0])
    # store the start and end indexes of the trajectory in the original data
    starts, ends = List([0]), List([0])
    data_num = dataset["rewards"].shape[0]
    print(f"Total number of data points: {data_num}")
    compute_cost_reward_return(
        dataset["rewards"], dataset["costs"], dataset["terminals"], dataset["timeouts"],
        returns, costs, starts, ends
    )
    return returns[1:], costs[1:], starts[1:], ends[1:]


def grid_filter(
    x,
    y,
    xmin=-np.inf,
    xmax=np.inf,
    ymin=-np.inf,
    ymax=np.inf,
    xbins=10,
    ybins=10,
    max_num_per_bin=10
):
    xmin, xmax = max(min(x), xmin), min(max(x), xmax)
    ymin, ymax = max(min(y), ymin), min(max(y), ymax)
    xbin_step = (xmax - xmin) / xbins
    ybin_step = (ymax - ymin) / ybins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(x)):
        if x[i] < xmin or x[i] > xmax or y[i] < ymin or y[i] > ymax:
            continue
        x_bin_idx = (x[i] - xmin) // xbin_step
        y_bin_idx = (y[i] - ymin) // ybin_step
        bin_hashmap[(x_bin_idx, y_bin_idx)].append(i)
    # start filtering
    indices = []
    for v in bin_hashmap.values():
        if len(v) > max_num_per_bin:
            # random sample max_num_per_bin indices
            indices += sample(v, max_num_per_bin)
        else:
            indices += v
    return indices


def filter_trajectory(
    cost,
    rew,
    traj,
    cost_min=-np.inf,
    cost_max=np.inf,
    rew_min=-np.inf,
    rew_max=np.inf,
    cost_bins=60,
    rew_bins=50,
    max_num_per_bin=1
):
    indices = grid_filter(
        cost,
        rew,
        cost_min,
        cost_max,
        rew_min,
        rew_max,
        xbins=cost_bins,
        ybins=rew_bins,
        max_num_per_bin=max_num_per_bin
    )
    cost2, rew2, traj2 = [], [], []
    for i in indices:
        cost2.append(cost[i])
        rew2.append(rew[i])
        traj2.append(traj[i])
    return cost2, rew2, traj2


def select_optimal_trajectory(cost, rew, traj, rmin=0, cost_bins=60, max_num_per_bin=1):

    xmin, xmax = min(cost), max(cost)
    xbin_step = (xmax - xmin) / cost_bins
    # the key is x y bin index, the value is a list of indices
    bin_hashmap = defaultdict(list)
    for i in range(len(cost)):
        if rew[i] < rmin:
            continue
        x_bin_idx = (cost[i] - xmin) // xbin_step
        bin_hashmap[x_bin_idx].append(i)

    # start filtering
    def sort_index(idx):
        return rew[idx]

    indices = []
    for v in bin_hashmap.values():
        idx = heapq.nlargest(max_num_per_bin, v, key=sort_index)
        indices += idx

    cost2, rew2, traj2 = [], [], []
    for i in indices:
        cost2.append(cost[i])
        rew2.append(rew[i])
        traj2.append(traj[i])
    return cost2, rew2, traj2

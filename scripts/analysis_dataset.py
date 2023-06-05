import argparse
import os
import os.path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from dsrl.generation.common import get_trajectory_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--output', '-o', type=str, default='cr-plot.png')
    parser.add_argument('--dir', '-d', type=str, default='fig')
    parser.add_argument('--maxlen', type=int, default=50000000)
    args = parser.parse_args()

    hfiles = []
    for file in args.files:
        hfiles.append(h5py.File(file, 'r'))

    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]

    print("*" * 10, "concatenating dataset from", "*" * 10)
    for file in args.files:
        print("*" * 10, file, "*" * 10)
    dataset_dict = {}
    for k in keys:
        d = [hfile[k] for hfile in hfiles]
        combined = np.concatenate(d, axis=0)[:args.maxlen]
        print(k, combined.shape)
        dataset_dict[k] = combined

    print("*" * 10, "dataset concatenation finished", "*" * 10)

    rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)

    print(f"Total number of trajectories: {len(rew_ret)}")

    plt.scatter(cost_ret, rew_ret)
    plt.xlabel("Costs")
    plt.ylabel("Rewards")
    output_path = osp.join(args.dir, args.output)
    if not osp.exists(args.dir):
        os.makedirs(args.dir)
    plt.savefig(output_path)

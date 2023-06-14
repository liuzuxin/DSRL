import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_trajectory_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', '-o', type=str, default='cr-plot.png')
    parser.add_argument('--maxlen', type=int, default=50000000)
    args = parser.parse_args()

    root_dir = args.root

    file_paths = glob.glob(os.path.join(root_dir, '**', 'dataset.hdf5'), recursive=True)

    for file_path in file_paths:
        dir_path = os.path.dirname(file_path)
        print("reading from ... ", dir_path)
        data = h5py.File(file_path, 'r')

        keys = [
            'observations', 'next_observations', 'actions', 'rewards', 'costs',
            'terminals', 'timeouts'
        ]

        dataset_dict = {}
        for k in keys:
            combined = np.array(data[k])[:args.maxlen]
            print(k, combined.shape)
            dataset_dict[k] = combined

        rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)

        print(f"Total number of trajectories: {len(rew_ret)}")

        plt.scatter(cost_ret, rew_ret)
        plt.xlabel("Costs")
        plt.ylabel("Rewards")
        output_path = os.path.join(dir_path, args.output)
        plt.savefig(output_path)
        plt.clf()

import argparse
from calendar import c
from collections import defaultdict
from fileinput import filename
import os
import os.path as osp
from sys import float_repr_style

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from dsrl.generation.common import get_trajectory_info, filter_trajectory, select_optimal_trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--output', '-o', type=str, default='car-circle')
    parser.add_argument('--dir', '-d', type=str, default='log')
    args = parser.parse_args()

    file_name = args.files[0].split("/")[-1].split(".")[0]
    print(file_name)

    hfiles = []
    for file in args.files:
        hfiles.append(h5py.File(file, 'r'))

    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]

    float_32_keys = ['observations', 'next_observations', 'actions', 'rewards', 'costs']

    print("*" * 10, "concatenating dataset from", "*" * 10)
    for file in args.files:
        print("*" * 10, file, "*" * 10)
    dataset_dict = {}
    for k in keys:
        d = [hfile[k] for hfile in hfiles]
        combined = np.concatenate(d, axis=0)
        if k in float_32_keys:
            combined = combined.astype(np.float32)
        print(k, combined.shape, combined.dtype)
        dataset_dict[k] = combined
    print("*" * 10, "dataset concatenation finished", "*" * 10)

    # process the dataset to the SDT format:
    # traj[i] is the i-th trajectory (a dict)
    rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)
    traj = []
    for i in trange(len(rew_ret), desc="Processing trajectories..."):
        start = start_index[i]
        end = end_index[i] + 1
        one_traj = {k: dataset_dict[k][start:end] for k in keys}
        traj.append(one_traj)
    print(f"Total number of trajectories: {len(traj)}")

    # process the trajectory data back to the d4rl data format:
    dataset = defaultdict(list)
    for d in traj:
        for k in keys:
            dataset[k].append(d[k])
    for k in keys:
        dataset[k] = np.squeeze(np.concatenate(dataset[k], axis=0))
        print(k, dataset[k].shape, dataset[k].dtype)

    # store the data
    if not osp.exists(args.dir):
        os.makedirs(args.dir)
    output_name = file_name + ".hdf5"
    output_file = osp.join(args.dir, output_name)
    outf = h5py.File(output_file, 'w')
    for k in keys:
        outf.create_dataset(k, data=dataset[k], compression='gzip')
    outf.close()

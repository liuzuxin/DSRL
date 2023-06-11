import argparse
import os

import gym
import dsrl.offline_metadrive
from dsrl.infos import DATASET_URLS

def get_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="Ant", help="road type, [Ant, Ball, Car, Drone]")
    parser.add_argument("--task", type=str, default="Run", help="[Run, Circle]")
    
    return parser
    
if __name__ == "__main__": 
    args = get_parser().parse_args()

    # make environments    
    env_name = args.agent + args.task
    print("Environment {} loaded: ".format(env_name))
    
    id = f'Offline{env_name}-v0'
    env = gym.make(id)
    
    
    # load dataset
    dataset = env.get_dataset()
    print("loaded data status: ",  env.observation_space.contains(dataset["observation"]))

    # interact with environment
    for _ in range(100): 
        obs, reward, terminal, truncate, info = env.step(env.action_space.sample())
    
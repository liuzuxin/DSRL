import argparse

import gym
import dsrl.offline_metadrive


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--road", type=str, default="easy", help="road type = [easy, medium, hard]"
    )
    parser.add_argument(
        "--traffic", type=str, default="sparse", help="traffic = [sparse, mean, dense]"
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # make environments
    env_name = args.road + args.traffic
    print("Environment {} loading...".format(env_name))

    id = f'OfflineMetadrive-{env_name}-v0'
    env = gym.make(id)

    # load dataset
    dataset = env.get_dataset()
    print(
        "loaded data status: ",
        env.observation_space.contains(dataset["observations"][0])
    )
    obs, info = env.reset()

    # interact with environment
    for _ in range(100):
        obs, reward, terminal, truncate, info = env.step(env.action_space.sample())

    print("done")
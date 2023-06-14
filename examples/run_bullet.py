import argparse

import gymnasium as gym
import dsrl.offline_bullet_safety_gym


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, default="Ant", help="agents = [Ant, Ball, Car, Drone]"
    )
    parser.add_argument("--task", type=str, default="Run", help="tasks = [Run, Circle]")

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
    print(
        "loaded data status: ",
        env.observation_space.contains(dataset["observations"][0])
    )
    obs, info = env.reset()

    # interact with environment
    for _ in range(100):
        obs, reward, terminal, truncate, info = env.step(env.action_space.sample())

    print("done")

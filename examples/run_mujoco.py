import argparse

import gymnasium as gym
import dsrl.offline_safety_gymnasium


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, default="Point", help="agents = [Point, Car]"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Goal1",
        help="tasks = [Circle1, Circle2, Goal1, Goal2, Button1, Button2, Push1, Push2]"
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # make environments
    env_name = args.agent + args.task
    print("Environment {} loading...".format(env_name))

    id = f'Offline{env_name}Gymnasium-v0'
    env = gym.make(id)

    # load dataset
    dataset = env.get_dataset()
    print(
        "loaded data status: ",
        env.observation_space.contains(dataset["observations"][0])
    )
    obs, info = env.reset()

    # interact with environment
    env.reset()
    for _ in range(100):
        obs, reward, terminal, truncate, info = env.step(env.action_space.sample())

    print("done")
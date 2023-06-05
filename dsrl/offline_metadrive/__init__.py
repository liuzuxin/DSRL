# from gymnasium.envs.registration import register, registry
from gym.envs.registration import register, registry
from dsrl import infos


for road in ["easy", "medium", "hard"]:
    for vehicle in ["sparse", "mean", "dense"]:
        env_name = road + vehicle
        register(
            id=f'OfflineMetadrive-{env_name}-v0',
            entry_point=f'dsrl.offline_metadrive.gym_envs:get_{env_name}_env',
            max_episode_steps=infos.DEFAULT_MAX_EPISODE_STEPS[env_name],
            kwargs={
                "dataset_url": infos.DATASET_URLS[env_name],
                "max_episode_reward": infos.MAX_EPISODE_REWARD[env_name],
                "min_episode_reward": infos.MIN_EPISODE_REWARD[env_name],
                "max_episode_cost": infos.MAX_EPISODE_COST[env_name],
                "min_episode_cost": infos.MIN_EPISODE_COST[env_name],
                }
        )

from safety_gymnasium.utils.registration import register
from gymnasium import register as gymnasium_register
from dsrl import infos


for agent in ['Point', 'Car']:
    for task in ['Circle1', 'Circle2',
                 'Goal1', 'Goal2',
                 'Button1', 'Button2',
                 'Push1', 'Push2']:
        env_name = agent + task
        try:
            register(
                id=f'Offline{env_name}-v0',
                entry_point=f'dsrl.offline_safety_gymnasium.gym_envs:get_{env_name}_env',
                max_episode_steps=infos.DEFAULT_MAX_EPISODE_STEPS[env_name],
                kwargs={
                    "dataset_url": infos.DATASET_URLS[env_name],
                    "max_episode_reward": infos.MAX_EPISODE_REWARD[env_name],
                    "min_episode_reward": infos.MIN_EPISODE_REWARD[env_name],
                    "max_episode_cost": infos.MAX_EPISODE_COST[env_name],
                    "min_episode_cost": infos.MIN_EPISODE_COST[env_name],
                    }
            )
            gymnasium_register(
                id=f'Offline{env_name}Gymnasium-v0',
                entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
                kwargs={'env_id': f'Offline{env_name}Gymnasium-v0'},
                max_episode_steps=infos.DEFAULT_MAX_EPISODE_STEPS[env_name],
            )
        except KeyError:
            print(f"env {env_name} not implemented yet")

for agent in ["Ant", "HalfCheetah", "Hopper", "Swimmer", "Walker2d"]:
    env_name = agent + "Velocity"
    register(
        id=f'Offline{env_name}-v1',
        entry_point=f'dsrl.offline_safety_gymnasium.gym_envs:get_{env_name}_env',
        max_episode_steps=infos.DEFAULT_MAX_EPISODE_STEPS[env_name],
        reward_threshold=infos.DEFAULT_REWARD_THRESHOLD[env_name],
        kwargs={
            "dataset_url": infos.DATASET_URLS[env_name],
            "max_episode_reward": infos.MAX_EPISODE_REWARD[env_name],
            "min_episode_reward": infos.MIN_EPISODE_REWARD[env_name],
            "max_episode_cost": infos.MAX_EPISODE_COST[env_name],
            "min_episode_cost": infos.MIN_EPISODE_COST[env_name],
            }
    )
    gymnasium_register(
        id=f'Offline{env_name}Gymnasium-v1',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'Offline{env_name}Gymnasium-v1'},
        max_episode_steps=infos.DEFAULT_MAX_EPISODE_STEPS[env_name],
        reward_threshold=infos.DEFAULT_REWARD_THRESHOLD[env_name]
    )

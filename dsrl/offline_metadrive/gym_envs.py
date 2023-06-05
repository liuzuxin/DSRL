import numpy as np
from metadrive.envs import SafeMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode
from .. import offline_env


class SafeMetaDriveEnv_FSRL(SafeMetaDriveEnv): 

    def default_config(self):
        config = super(SafeMetaDriveEnv_FSRL, self).default_config()
        config.update(
            {
                "idm_target_speed": 30,
                "idm_acc_factor": 1.0, 
                "idm_deacc_factor": 5.0, 
                "crash_vehicle_penalty": 0.0,
                "random_traffic": True,
                "traffic_mode": TrafficMode.Hybrid
            },
            allow_add_new_key=True
        )
        return config

    def step(self, actions):
        o, r, d, i = super(SafeMetaDriveEnv_FSRL, self).step(actions)
        i["velocity_cost"] = max(0, 1e-2*(i["velocity"]-10.))
        i["cost"] += i["velocity_cost"]
        i["proximity_cost"] = 1e-2*max(0, 0.1-np.min(o[-240:]))**2
        i["cost"] += i["proximity_cost"]
        
        i["out_of_road_cost"] = self.config["out_of_road_cost"] if i["out_of_road"] else 0
        i["crash_cost"] = self.config["crash_vehicle_cost"] if i["crash"] else 0
        
        if i["max_step"]: 
            truncated = True
        else: 
            truncated = False
        if i["arrive_dest"] or i["crash"] or i["out_of_road"]: 
            terminated = True
        else: 
            terminated = False
        return o, r, terminated, truncated, i

    def reset(self, seed=None, options=None):
        return super().reset(force_seed=seed), {}


class EasySparseEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 0,
            "traffic_density": 0.1, 
            "map": 3, 
            "map_config": {"type": "block_sequence", "config": "SCS"}, 
            "accident_prob": 0.0,
            "environment_num": 1, 
            "horizon": 1000,
        }
        super().__init__(config)

class EasyMeanEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 0, 
            "traffic_density": 0.15,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "SCS"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class EasyDenseEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 0, 
            "traffic_density": 0.2,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "SCS"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class MediumSparseEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 100, 
            "traffic_density": 0.1,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "XST"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class MediumMeanEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 100, 
            "traffic_density": 0.15,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "XST"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class MediumDenseEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 100, 
            "traffic_density": 0.2,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "XST"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class HardSparseEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 200, 
            "traffic_density": 0.1,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "TRO"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class HardMeanEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 200, 
            "traffic_density": 0.15,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "TRO"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class HardDenseEnv(SafeMetaDriveEnv_FSRL):
    def __init__(self):
        config = {
            "start_seed": 200, 
            "traffic_density": 0.2,
            "map": 3,
            "map_config": {"type": "block_sequence", "config": "TRO"}, 
            "accident_prob": 0.0,
            "environment_num": 1,
            "horizon": 1000,
        }
        super().__init__(config)

class OfflineEasySparseEnv(EasySparseEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        EasySparseEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineEasyMeanEnv(EasyMeanEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        EasyMeanEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineEasyDenseEnv(EasyDenseEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        EasyDenseEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineMediumSparseEnv(MediumSparseEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        MediumSparseEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineMediumMeanEnv(MediumMeanEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        MediumMeanEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineMediumDenseEnv(MediumDenseEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        MediumDenseEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHardSparseEnv(HardSparseEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HardSparseEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHardMeanEnv(HardMeanEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HardMeanEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHardDenseEnv(HardDenseEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HardDenseEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

def get_easysparse_env(**kwargs):
    return OfflineEasySparseEnv(**kwargs)

def get_easymean_env(**kwargs):
    return OfflineEasyMeanEnv(**kwargs)

def get_easydense_env(**kwargs):
    return OfflineEasyDenseEnv(**kwargs)

def get_mediumsparse_env(**kwargs):
    return OfflineMediumSparseEnv(**kwargs)

def get_mediummean_env(**kwargs):
    return OfflineMediumMeanEnv(**kwargs)

def get_mediumdense_env(**kwargs):
    return OfflineMediumDenseEnv(**kwargs)

def get_hardsparse_env(**kwargs):
    return OfflineHardSparseEnv(**kwargs)

def get_hardmean_env(**kwargs):
    return OfflineHardMeanEnv(**kwargs)

def get_harddense_env(**kwargs):
    return OfflineHardDenseEnv(**kwargs)


from .. import offline_env
from safety_gymnasium.builder import Builder
from safety_gymnasium.tasks.safety_velocity.safety_ant_velocity_v1 import SafetyAntVelocityEnv
from safety_gymnasium.tasks.safety_velocity.safety_half_cheetah_velocity_v1 import SafetyHalfCheetahVelocityEnv
from safety_gymnasium.tasks.safety_velocity.safety_hopper_velocity_v1 import SafetyHopperVelocityEnv
from safety_gymnasium.tasks.safety_velocity.safety_swimmer_velocity_v1 import SafetySwimmerVelocityEnv
from safety_gymnasium.tasks.safety_velocity.safety_walker2d_velocity_v1 import SafetyWalker2dVelocityEnv

# env configs from https://github.com/OmniSafeAI/safety-gymnasium/tree/main
class CarCircle1Env(Builder):
    def __init__(self):
        task_id = "SafetyCarCircle1-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class CarCircle2Env(Builder):
    def __init__(self):
        task_id = "SafetyCarCircle2-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class CarGoal1Env(Builder):
    def __init__(self):
        task_id = "SafetyCarGoal1-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class CarGoal2Env(Builder):
    def __init__(self):
        task_id = "SafetyCarGoal2-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)
        
class CarButton1Env(Builder):
    def __init__(self):
        task_id = "SafetyCarButton1-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class CarButton2Env(Builder):
    def __init__(self):
        task_id = "SafetyCarButton2-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class CarPush1Env(Builder):
    def __init__(self):
        task_id = "SafetyCarPush1-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class CarPush2Env(Builder):
    def __init__(self):
        task_id = "SafetyCarPush2-v0"
        config = {"agent_name": "Car"}
        super().__init__(task_id, config)

class PointCircle1Env(Builder):
    def __init__(self):
        task_id = "SafetyPointCircle1-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)

class PointCircle2Env(Builder):
    def __init__(self):
        task_id = "SafetyPointCircle2-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)

class PointGoal1Env(Builder):
    def __init__(self):
        task_id = "SafetyPointGoal1-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)

class PointGoal2Env(Builder):
    def __init__(self):
        task_id = "SafetyPointGoal2-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)

class PointButton1Env(Builder):
    def __init__(self):
        task_id = "SafetyPointButton1-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)
        
class PointButton2Env(Builder):
    def __init__(self):
        task_id = "SafetyPointButton2-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)
        
class PointPush1Env(Builder):
    def __init__(self):
        task_id = "SafetyPointPush1-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)
        
class PointPush2Env(Builder):
    def __init__(self):
        task_id = "SafetyPointPush2-v0"
        config = {"agent_name": "Point"}
        super().__init__(task_id, config)

class OfflineCarCircle1Env(CarCircle1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarCircle1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineCarCircle2Env(CarCircle2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarCircle2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)
        
class OfflineCarGoal1Env(CarGoal1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarGoal1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineCarGoal2Env(CarGoal2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarGoal2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)
        
class OfflineCarButton1Env(CarButton1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarButton1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineCarButton2Env(CarButton2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarButton2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)
        
class OfflineCarPush1Env(CarPush1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarPush1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineCarPush2Env(CarPush2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarPush2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointCircle1Env(PointCircle1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointCircle1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointCircle2Env(PointCircle2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointCircle2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointGoal1Env(PointGoal1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointGoal1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointGoal2Env(PointGoal2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointGoal2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointButton1Env(PointButton1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointButton1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointButton2Env(PointButton2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointButton2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointPush1Env(PointPush1Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointPush1Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflinePointPush2Env(PointPush2Env, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        PointPush2Env.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineAntVelocityEnv(SafetyAntVelocityEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SafetyAntVelocityEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHalfCheetahVelocityEnv(SafetyHalfCheetahVelocityEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SafetyHalfCheetahVelocityEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHopperVelocityEnv(SafetyHopperVelocityEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SafetyHopperVelocityEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineSwimmerVelocityEnv(SafetySwimmerVelocityEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SafetySwimmerVelocityEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineWalker2dVelocityEnv(SafetyWalker2dVelocityEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SafetyWalker2dVelocityEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

def get_CarCircle1_env(**kwargs):
    return OfflineCarCircle1Env(**kwargs)

def get_CarCircle2_env(**kwargs):
    return OfflineCarCircle2Env(**kwargs)

def get_CarGoal1_env(**kwargs):
    return OfflineCarGoal1Env(**kwargs)

def get_CarGoal2_env(**kwargs):
    return OfflineCarGoal2Env(**kwargs)

def get_CarButton1_env(**kwargs):
    return OfflineCarButton1Env(**kwargs)

def get_CarButton2_env(**kwargs):
    return OfflineCarButton2Env(**kwargs)

def get_CarPush1_env(**kwargs):
    return OfflineCarPush1Env(**kwargs)

def get_CarPush2_env(**kwargs):
    return OfflineCarPush2Env(**kwargs)

def get_PointCircle1_env(**kwargs):
    return OfflinePointCircle1Env(**kwargs)

def get_PointCircle2_env(**kwargs):
    return OfflinePointCircle2Env(**kwargs)

def get_PointGoal1_env(**kwargs):
    return OfflinePointGoal1Env(**kwargs)

def get_PointGoal2_env(**kwargs):
    return OfflinePointGoal2Env(**kwargs)

def get_PointButton1_env(**kwargs):
    return OfflinePointButton1Env(**kwargs)

def get_PointButton2_env(**kwargs):
    return OfflinePointButton2Env(**kwargs)

def get_PointPush1_env(**kwargs):
    return OfflinePointPush1Env(**kwargs)

def get_PointPush2_env(**kwargs):
    return OfflinePointPush2Env(**kwargs)

def get_AntVelocity_env(**kwargs):
    return OfflineAntVelocityEnv(**kwargs)

def get_HalfCheetahVelocity_env(**kwargs):
    return OfflineHalfCheetahVelocityEnv(**kwargs)

def get_HopperVelocity_env(**kwargs):
    return OfflineHopperVelocityEnv(**kwargs)

def get_SwimmerVelocity_env(**kwargs):
    return OfflineSwimmerVelocityEnv(**kwargs)

def get_Walker2dVelocity_env(**kwargs):
    return OfflineWalker2dVelocityEnv(**kwargs)

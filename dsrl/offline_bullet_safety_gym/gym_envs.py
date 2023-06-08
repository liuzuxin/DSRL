from .. import offline_env
from bullet_safety_gym.envs.builder import EnvironmentBuilder


# env configs from https://github.com/liuzuxin/Bullet-Safety-Gym
class AntCircleEnv(EnvironmentBuilder):
    def __init__(self):
        agent='Ant'
        task='CircleTask'
        obstacles={}
        world={'name': 'Octagon'}
        super().__init__(agent, task, obstacles, world)

class AntRunEnv(EnvironmentBuilder):
    def __init__(self):
        agent = "Ant"
        task = "RunTask"
        obstacles = {}
        world = {"name": "Plane200", "factor": 1}
        super().__init__(agent, task, obstacles, world)

class BallCircleEnv(EnvironmentBuilder):
    def __init__(self):
        agent = 'Ball'
        task = 'CircleTask'
        obstacles = {}
        world = {'name': 'Octagon'}
        super().__init__(agent, task, obstacles, world)

class BallRunEnv(EnvironmentBuilder):
    def __init__(self):
        agent = 'Ball'
        task = 'RunTask'
        obstacles = {}
        world = {'name': 'Plane200', 'factor': 1}
        super().__init__(agent, task, obstacles, world)

class CarCircleEnv(EnvironmentBuilder):
    def __init__(self):
        agent = "RaceCar"
        task = "CircleTask"
        obstacles = {}
        world = {"name": "Octagon"}
        super().__init__(agent, task, obstacles, world)

class CarRunEnv(EnvironmentBuilder):
    def __init__(self):
        agent = 'RaceCar'
        task = 'RunTask'
        obstacles = {}
        world = {'name': 'Plane200', 'factor': 1}
        super().__init__(agent, task, obstacles, world)

class DroneCircleEnv(EnvironmentBuilder):
    def __init__(self):
        agent = 'Drone'
        task = 'CircleTask'
        obstacles = {}
        world = {'name': 'Octagon'}
        super().__init__(agent, task, obstacles, world)

class DroneRunEnv(EnvironmentBuilder):
    def __init__(self):
        agent='Drone'
        task='RunTask'
        obstacles={}
        world={'name': 'Plane200', 'factor': 1}
        super().__init__(agent, task, obstacles, world)

class OfflineAntCircleEnv(AntCircleEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntCircleEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineAntRunEnv(AntRunEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntRunEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineBallCircleEnv(BallCircleEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        BallCircleEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineBallRunEnv(BallRunEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        BallRunEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineCarCircleEnv(CarCircleEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarCircleEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineCarRunEnv(CarRunEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        CarRunEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineDroneCircleEnv(DroneCircleEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        DroneCircleEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineDroneRunEnv(DroneRunEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        DroneRunEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

def get_AntCircle_env(**kwargs):
    return OfflineAntCircleEnv(**kwargs)

def get_AntRun_env(**kwargs):
    return OfflineAntRunEnv(**kwargs)

def get_BallCircle_env(**kwargs):
    return OfflineBallCircleEnv(**kwargs)

def get_BallRun_env(**kwargs):
    return OfflineBallRunEnv(**kwargs)

def get_CarCircle_env(**kwargs):
    return OfflineCarCircleEnv(**kwargs)

def get_CarRun_env(**kwargs):
    return OfflineCarRunEnv(**kwargs)

def get_DroneCircle_env(**kwargs):
    return OfflineDroneCircleEnv(**kwargs)

def get_DroneRun_env(**kwargs):
    return OfflineDroneRunEnv(**kwargs)

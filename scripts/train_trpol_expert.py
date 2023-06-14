import os
import signal
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import bullet_safety_gym
import gymnasium as gym
import numpy as np
import pyrallis
import safety_gymnasium
import torch
import torch.nn as nn
from fsrl.data import BasicCollector, FastCollector, TrajectoryBuffer
from fsrl.policy import TRPOLagrangian
from fsrl.trainer import OnpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyCarCircle-v0"
    cost_start: float = 5
    cost_end: float = 100
    epoch_start: int = 100
    epoch_end: int = 900
    epoch: int = 1000
    max_traj_len: int = 1500
    collect_in_train: bool = True
    # for trajectory buffer
    rmin: float = -9999
    rmax: float = 9999
    cmin: float = 0
    cmax: float = 300
    use_lagrangian: bool = True
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 10
    use_default_cfg: bool = True
    # algorithm params
    lr: float = 5e-4
    hidden_sizes: Tuple[int, ...] = (128, 128)
    unbounded: bool = False
    last_layer_scale: bool = False
    # PPO specific arguments
    target_kl: float = 0.001
    backtrack_coeff: float = 0.8
    max_backtracks: int = 10
    optim_critic_iters: int = 20
    gae_lambda: float = 0.95
    norm_adv: bool = True  # good for improving training stability
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.99
    max_batchsize: int = 100000
    rew_norm: bool = False
    deterministic_eval: bool = False
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    episode_per_collect: int = 10
    step_per_epoch: int = 10000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 20
    testing_num: int = 2
    # general params
    # batch-size >> steps per collect means calculating all data in one singe forward.
    batch_size: int = 99999
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "dsrl_expert"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "trpol"
    suffix: Optional[str] = ""


@dataclass
class BulletCarCircleCfg(TrainCfg):
    pass


@dataclass
class BulletBallCircleCfg(TrainCfg):
    task: str = "SafetyBallCircle-v0"
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 100
    epoch_end: int = 900
    epoch: int = 1000
    max_traj_len: int = 1000
    collect_in_train: bool = True
    testing_num: int = 2
    deterministic_eval: bool = False


@dataclass
class BulletAntRunCfg(TrainCfg):
    task: str = "SafetyAntRun-v0"
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 400
    epoch_end: int = 2500
    epoch: int = 2600
    max_traj_len: int = 2000
    collect_in_train: bool = False
    testing_num: int = 4
    deterministic_eval: bool = False


@dataclass
class BulletAntCircleCfg(TrainCfg):
    task: str = "SafetyAntCircle-v0"
    hidden_sizes = [256, 256]
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 2000
    epoch_end: int = 5000
    epoch: int = 5400
    max_traj_len: int = 5000
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False


@dataclass
class BulletDroneRunCfg(TrainCfg):
    task: str = "SafetyDroneRun-v0"
    cost_start: float = 100
    cost_end: float = 5
    epoch_start: int = 50
    epoch_end: int = 800
    epoch: int = 1000
    max_traj_len: int = 1500
    collect_in_train: bool = True
    testing_num: int = 2
    deterministic_eval: bool = False
    target_kl: float = 0.0005


@dataclass
class BulletDroneCircleCfg(TrainCfg):
    task: str = "SafetyDroneCircle-v0"
    cost_start: float = 5
    cost_end: float = 150
    epoch_start: int = 500
    epoch_end: int = 2500
    epoch: int = 2600
    max_traj_len: int = 2000
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False


@dataclass
class MujocoPointCircle1Cfg(TrainCfg):
    task: str = "SafetyPointCircle1Gymnasium-v0"
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 80
    epoch_end: int = 500
    epoch: int = 600
    max_traj_len: int = 1000
    collect_in_train: bool = False
    testing_num: int = 4
    deterministic_eval: bool = False


@dataclass
class MujocoPointCircle2Cfg(TrainCfg):
    task: str = "SafetyPointCircle2Gymnasium-v0"
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 80
    epoch_end: int = 900
    epoch: int = 1000
    max_traj_len: int = 1000
    collect_in_train: bool = False
    testing_num: int = 4
    deterministic_eval: bool = False


@dataclass
class MujocoPointGoal1Cfg(TrainCfg):
    task: str = "SafetyPointGoal1Gymnasium-v0"
    cost_start: float = 80
    cost_end: float = 5
    epoch_start: int = 200
    epoch_end: int = 1500
    epoch: int = 1800
    max_traj_len: int = 1500
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 100


@dataclass
class MujocoPointGoal2Cfg(TrainCfg):
    task: str = "SafetyPointGoal2Gymnasium-v0"
    cost_start: float = 180
    cost_end: float = 5
    epoch_start: int = 400
    epoch_end: int = 2000
    epoch: int = 2200
    max_traj_len: int = 3000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 200


@dataclass
class MujocoPointButton1Cfg(TrainCfg):
    task: str = "SafetyPointButton1Gymnasium-v0"
    cost_start: float = 150
    cost_end: float = 5
    epoch_start: int = 200
    epoch_end: int = 1800
    epoch: int = 2000
    max_traj_len: int = 2000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 200


@dataclass
class MujocoPointButton2Cfg(TrainCfg):
    task: str = "SafetyPointButton2Gymnasium-v0"
    cost_start: float = 200
    cost_end: float = 5
    epoch_start: int = 800
    epoch_end: int = 3000
    epoch: int = 3200
    max_traj_len: int = 3000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 250


@dataclass
class MujocoPointPush1Cfg(TrainCfg):
    task: str = "SafetyPointPush1Gymnasium-v0"
    cost_start: float = 100
    cost_end: float = 5
    epoch_start: int = 400
    epoch_end: int = 2500
    epoch: int = 2800
    max_traj_len: int = 2000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 150


@dataclass
class MujocoPointPush2Cfg(TrainCfg):
    task: str = "SafetyPointPush2Gymnasium-v0"
    cost_start: float = 150
    cost_end: float = 5
    epoch_start: int = 500
    epoch_end: int = 2500
    epoch: int = 2800
    max_traj_len: int = 3000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 200


@dataclass
class MujocoCarCircle1Cfg(TrainCfg):
    task: str = "SafetyCarCircle1Gymnasium-v0"
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 80
    epoch_end: int = 500
    epoch: int = 600
    max_traj_len: int = 1000
    collect_in_train: bool = False
    testing_num: int = 4
    deterministic_eval: bool = False


@dataclass
class MujocoCarCircle2Cfg(TrainCfg):
    task: str = "SafetyCarCircle2Gymnasium-v0"
    cost_start: float = 5
    cost_end: float = 200
    epoch_start: int = 80
    epoch_end: int = 900
    epoch: int = 1000
    max_traj_len: int = 1000
    collect_in_train: bool = False
    testing_num: int = 4
    deterministic_eval: bool = False


@dataclass
class MujocoCarGoal1Cfg(TrainCfg):
    task: str = "SafetyCarGoal1Gymnasium-v0"
    cost_start: float = 80
    cost_end: float = 5
    epoch_start: int = 200
    epoch_end: int = 1000
    epoch: int = 1200
    max_traj_len: int = 1500
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 120


@dataclass
class MujocoCarGoal2Cfg(TrainCfg):
    task: str = "SafetyCarGoal2Gymnasium-v0"
    cost_start: float = 180
    cost_end: float = 5
    epoch_start: int = 400
    epoch_end: int = 2000
    epoch: int = 2200
    max_traj_len: int = 3000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 200


@dataclass
class MujocoCarButton1Cfg(TrainCfg):
    task: str = "SafetyCarButton1Gymnasium-v0"
    cost_start: float = 200
    cost_end: float = 10
    epoch_start: int = 400
    epoch_end: int = 2000
    epoch: int = 2200
    max_traj_len: int = 2500
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 250


@dataclass
class MujocoCarButton2Cfg(TrainCfg):
    task: str = "SafetyCarButton2Gymnasium-v0"
    cost_start: float = 300
    cost_end: float = 10
    epoch_start: int = 400
    epoch_end: int = 3000
    epoch: int = 3500
    max_traj_len: int = 3000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 300


@dataclass
class MujocoCarPush1Cfg(TrainCfg):
    task: str = "SafetyCarPush1Gymnasium-v0"
    cost_start: float = 150
    cost_end: float = 10
    epoch_start: int = 500
    epoch_end: int = 3500
    epoch: int = 4000
    max_traj_len: int = 2000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 200


@dataclass
class MujocoCarPush2Cfg(TrainCfg):
    task: str = "SafetyCarPush2Gymnasium-v0"
    cost_start: float = 200
    cost_end: float = 10
    epoch_start: int = 500
    epoch_end: int = 3500
    epoch: int = 4000
    max_traj_len: int = 3000
    collect_in_train: bool = True
    testing_num: int = 1
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 250


@dataclass
class MujocoHalfCheetahCfg(TrainCfg):
    task: str = "SafetyHalfCheetahVelocityGymnasium-v1"
    cost_start: float = 250
    cost_end: float = 10
    epoch_start: int = 800
    epoch_end: int = 2800
    epoch: int = 3000
    max_traj_len: int = 1500
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 250
    gamma: float = 0.995


@dataclass
class MujocoHopperCfg(TrainCfg):
    task: str = "SafetyHopperVelocityGymnasium-v1"
    cost_start: float = 250
    cost_end: float = 10
    epoch_start: int = 800
    epoch_end: int = 2500
    epoch: int = 3000
    max_traj_len: int = 1500
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 250
    gamma: float = 0.995


@dataclass
class MujocoSwimmerCfg(TrainCfg):
    task: str = "SafetySwimmerVelocityGymnasium-v1"
    cost_start: float = 200
    cost_end: float = 10
    epoch_start: int = 800
    epoch_end: int = 2500
    epoch: int = 3000
    max_traj_len: int = 1500
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 200
    gamma: float = 0.995


@dataclass
class MujocoWalker2dCfg(TrainCfg):
    task: str = "SafetyWalker2dVelocityGymnasium-v1"
    cost_start: float = 10
    cost_end: float = 300
    epoch_start: int = 1500
    epoch_end: int = 4000
    epoch: int = 4200
    max_traj_len: int = 2000
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False
    hidden_sizes: Tuple[int, ...] = (256, 256)
    last_layer_scale: bool = True
    # PPO specific arguments
    target_kl: float = 0.005
    gamma: float = 0.995
    rmin: float = 0
    cmax: float = 250


@dataclass
class MujocoAntCfg(TrainCfg):
    task: str = "SafetyAntVelocityGymnasium-v1"
    cost_start: float = 20
    cost_end: float = 300
    epoch_start: int = 1000
    epoch_end: int = 3000
    epoch: int = 3500
    hidden_sizes: Tuple[int, ...] = (256, 256)
    max_traj_len: int = 2000
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False
    rmin: float = 0
    cmax: float = 250


@dataclass
class MujocoHumanoidCfg(TrainCfg):
    task: str = "SafetyHumanoidVelocityGymnasium-v1"
    cost_start: float = 300
    cost_end: float = 10
    epoch_start: int = 1000
    epoch_end: int = 6500
    epoch: int = 7000
    max_traj_len: int = 3000
    collect_in_train: bool = False
    testing_num: int = 2
    deterministic_eval: bool = False
    hidden_sizes: Tuple[int, ...] = (256, 256)
    target_kl: float = 0.005
    norm_adv: bool = False
    unbounded = True


TASK_TO_CFG = {
    "SafetyCarRun-v0": TrainCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyBallRun-v0": TrainCfg,
    "SafetyBallCircle-v0": BulletBallCircleCfg,
    "SafetyDroneRun-v0": BulletDroneRunCfg,
    "SafetyDroneCircle-v0": BulletDroneCircleCfg,
    "SafetyAntRun-v0": BulletAntRunCfg,
    "SafetyAntCircle-v0": BulletAntCircleCfg,
    "SafetyPointCircle1Gymnasium-v0": MujocoPointCircle1Cfg,
    "SafetyPointCircle2Gymnasium-v0": MujocoPointCircle2Cfg,
    "SafetyPointGoal1Gymnasium-v0": MujocoPointGoal1Cfg,
    "SafetyPointGoal2Gymnasium-v0": MujocoPointGoal2Cfg,
    "SafetyPointButton1Gymnasium-v0": MujocoPointButton1Cfg,
    "SafetyPointButton2Gymnasium-v0": MujocoPointButton2Cfg,
    "SafetyPointPush1Gymnasium-v0": MujocoPointPush1Cfg,
    "SafetyPointPush2Gymnasium-v0": MujocoPointPush2Cfg,
    "SafetyCarCircle1Gymnasium-v0": MujocoCarCircle1Cfg,
    "SafetyCarCircle2Gymnasium-v0": MujocoCarCircle2Cfg,
    "SafetyCarGoal1Gymnasium-v0": MujocoCarGoal1Cfg,
    "SafetyCarGoal2Gymnasium-v0": MujocoCarGoal2Cfg,
    "SafetyCarButton1Gymnasium-v0": MujocoCarButton1Cfg,
    "SafetyCarButton2Gymnasium-v0": MujocoCarButton2Cfg,
    "SafetyCarPush1Gymnasium-v0": MujocoCarPush1Cfg,
    "SafetyCarPush2Gymnasium-v0": MujocoCarPush2Cfg,
    "SafetyHalfCheetahVelocityGymnasium-v1": MujocoHalfCheetahCfg,
    "SafetyHopperVelocityGymnasium-v1": MujocoHopperCfg,
    "SafetySwimmerVelocityGymnasium-v1": MujocoSwimmerCfg,
    "SafetyWalker2dVelocityGymnasium-v1": MujocoWalker2dCfg,
    "SafetyAntVelocityGymnasium-v1": MujocoAntCfg,
    "SafetyHumanoidVelocityGymnasium-v1": MujocoHumanoidCfg,
}


class ActorProbLargeVar(ActorProb):

    SIGMA_MIN = -3
    SIGMA_MAX = 2

    def forward(
        self,
        obs,
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=self.SIGMA_MIN, max=self.SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


def cost_limit_scheduler(epoch, epoch_start, epoch_end, cost_start, cost_end):
    x = min(max(0, epoch - epoch_start), epoch_end - epoch_start)
    cost = cost_start - x * (cost_start - cost_end) / (epoch_end - epoch_start)
    return cost


@pyrallis.wrap()
def train(args: TrainCfg):
    # set seed and computing
    seed_all(args.seed)
    if args.device == "cpu":
        print("Using cpu with %i threads." % args.thread)
        torch.set_num_threads(args.thread)

    task = args.task
    default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else TrainCfg()

    ######### comment the following line for customized configs!!! #############
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        args = default_cfg
        args.task = task

    # logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_start)
                                                ) + "-" + str(int(args.cost_end))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    #logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # model
    env = gym.make(args.task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProbLargeVar(
        net,
        action_shape,
        max_action=max_action,
        unbounded=args.unbounded,
        device=args.device
    ).to(args.device)
    critic = [
        Critic(
            Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
            device=args.device
        ).to(args.device) for _ in range(2)
    ]
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    if args.last_layer_scale:
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = TRPOLagrangian(
        actor,
        critic,
        optim,
        dist,
        logger=logger,
        target_kl=args.target_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks,
        optim_critic_iters=args.optim_critic_iters,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.norm_adv,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_start,
        rescaling=args.rescaling,
        gamma=args.gamma,
        max_batchsize=args.max_batchsize,
        reward_normalization=args.rew_norm,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_scheduler=None
    )

    # collector
    traj_buffer = TrajectoryBuffer(
        args.max_traj_len,
        filter_interval=1.5,
        rmin=args.rmin,
        rmax=args.rmax,
        cmin=args.cmin,
        cmax=args.cmax
    )
    if args.collect_in_train:
        train_collector = BasicCollector(
            policy, env, ReplayBuffer(args.buffer_size), traj_buffer=traj_buffer
        )
    else:
        training_num = min(args.training_num, args.episode_per_collect)
        worker = eval(args.worker)
        train_envs = worker([lambda: gym.make(args.task) for _ in range(training_num)])
        train_collector = FastCollector(
            policy,
            train_envs,
            VectorReplayBuffer(args.buffer_size, len(train_envs)),
            exploration_noise=True,
        )

    test_collector = BasicCollector(policy, gym.make(args.task), traj_buffer=traj_buffer)

    def stop_fn(reward, cost):
        return False

    # def checkpoint_fn():
    #     return {"model": policy.state_dict()}
    # if args.save_ckpt:
    #     logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_end,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )

    def saving_dataset():
        traj_buffer.save(logger.log_dir)

    def term_handler(signum, frame):
        print("Sig term handler, saving the dataset...")
        saving_dataset()
        sys.exit(0)

    signal.signal(signal.SIGTERM, term_handler)

    try:
        for epoch, epoch_stat, info in trainer:
            # print(f"Epoch: {epoch}")
            # print(info)
            print(f"Trajs: {len(traj_buffer.buffer)}, transitions: {len(traj_buffer)}")
            cost = cost_limit_scheduler(
                epoch, args.epoch_start, args.epoch_end, args.cost_start, args.cost_end
            )
            policy.update_cost_limit(cost)
            logger.store(tab="train", cost_limit=cost, epoch=epoch)
    except KeyboardInterrupt:
        print("keyboardinterrupt detected, saving the dataset...")
        saving_dataset()
    except Exception as e:
        print("exception catched, saving the dataset...")
        saving_dataset()

    saving_dataset()


if __name__ == "__main__":
    train()

import os
import os.path as osp
import pprint
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import bullet_safety_gym
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from tianshou.data import VectorReplayBuffer, ReplayBuffer
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.data import FastCollector, BasicCollector, TrajectoryBuffer
from fsrl.policy import PPOLagrangian
from fsrl.trainer import OnpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


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
    use_lagrangian: bool = True
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 10
    # algorithm params
    lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    # PPO specific arguments
    target_kl: float = 0.02
    vf_coef: float = 0.25
    max_grad_norm: Optional[float] = 0.5
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    dual_clip: Optional[float] = None
    value_clip: bool = False  # no need
    norm_adv: bool = True  # good for improving training stability
    recompute_adv: bool = False
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.98
    max_batchsize: int = 100000
    rew_norm: bool = False  # no need, it will slow down training and decrease final perf
    recompute_adv: bool = False
    deterministic_eval: bool = False
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    episode_per_collect: int = 20
    step_per_epoch: int = 20000
    repeat_per_collect: int = 4  # increasing this can improve efficiency, but less stability
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 20
    testing_num: int = 2
    # general params
    batch_size: int = 512
    reward_threshold: float = 1000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = False  # set this to True to save the policy model
    verbose: bool = False
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "dsrl-collect"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "ppol"
    suffix: Optional[str] = "v1"


########################################################
######## bullet-safety-gym task default configs ########
########################################################


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
    epoch_end: int = 1000
    epoch: int = 1200
    max_traj_len: int = 1500
    collect_in_train: bool = False
    testing_num: int = 5
    deterministic_eval: bool = False
    target_kl: float = 0.005
    max_grad_norm: Optional[float] = 0.2
    eps_clip: float = 0.05


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


TASK_TO_CFG = {
    "SafetyCarRun-v0": TrainCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyBallRun-v0": TrainCfg,
    "SafetyBallCircle-v0": BulletBallCircleCfg,
    "SafetyDroneRun-v0": BulletDroneRunCfg,
    "SafetyDroneCircle-v0": BulletDroneCircleCfg,
    "SafetyAntRun-v0": BulletAntRunCfg,
    "SafetyAntCircle-v0": BulletAntCircleCfg,
}


class ActorProbLargeVar(ActorProb):

    SIGMA_MIN = -1.5
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
            sigma = torch.clamp(self.sigma(logits),
                                min=self.SIGMA_MIN,
                                max=self.SIGMA_MAX).exp()
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
    args = default_cfg

    default_cfg = asdict(default_cfg)

    # logger
    cfg = asdict(args)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_start)) + "-" + str(
            int(args.cost_end))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    #logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # model
    env = gym.make(args.task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProbLargeVar(net,
                              action_shape,
                              max_action=max_action,
                              device=args.device).to(args.device)
    critic = [
        Critic(Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
               device=args.device).to(args.device) for _ in range(2)
    ]
    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOLagrangian(actor,
                           critic,
                           optim,
                           dist,
                           logger=logger,
                           target_kl=args.target_kl,
                           vf_coef=args.vf_coef,
                           max_grad_norm=args.max_grad_norm,
                           gae_lambda=args.gae_lambda,
                           eps_clip=args.eps_clip,
                           dual_clip=args.dual_clip,
                           value_clip=args.value_clip,
                           advantage_normalization=args.norm_adv,
                           recompute_advantage=args.recompute_adv,
                           use_lagrangian=args.use_lagrangian,
                           lagrangian_pid=args.lagrangian_pid,
                           cost_limit=args.cost_start,
                           rescaling=args.rescaling,
                           gamma=args.gamma,
                           max_batchsize=args.max_batchsize,
                           reward_normalization=args.norm_adv,
                           deterministic_eval=args.deterministic_eval,
                           action_scaling=args.action_scaling,
                           action_bound_method=args.action_bound_method)

    # collector
    traj_buffer = TrajectoryBuffer(args.max_traj_len, filter_interval=1.5)
    if args.collect_in_train:
        train_collector = BasicCollector(policy,
                                         env,
                                         ReplayBuffer(args.buffer_size),
                                         traj_buffer=traj_buffer)
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

    for epoch, epoch_stat, info in trainer:
        # print(f"Epoch: {epoch}")
        # print(info)
        print(f"Trajs: {len(traj_buffer.buffer)}, transitions: {len(traj_buffer)}")
        cost = cost_limit_scheduler(epoch, args.epoch_start, args.epoch_end,
                                    args.cost_start, args.cost_end)
        policy.update_cost_limit(cost)
        logger.store(tab="train", cost_limit=cost, epoch=epoch)

    traj_buffer.save(args.logdir)


if __name__ == "__main__":
    train()

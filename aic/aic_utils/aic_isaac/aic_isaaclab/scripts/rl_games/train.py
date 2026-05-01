# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games (AIC fork: registers local `aic_task` Gym envs)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--video_upload_every",
    type=int,
    default=2,
    help="Upload one video to wandb every N training epochs (0 = never upload). Default: 2.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import logging
import math
import os
import random
import time
from datetime import datetime

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import aic_task.tasks  # noqa: F401 — register AIC-* Gym envs

# import logger
logger = logging.getLogger(__name__)


class AicWandbObserver(IsaacAlgoObserver):
    """IsaacAlgoObserver extended with rate-limited wandb video uploads.

    Uploads the latest recorded video to wandb every ``upload_every_n_epochs``
    training epochs and deletes older local copies to save disk space.
    Reward/metric logging is handled automatically via tensorboard sync.
    """

    def __init__(self, video_folder: str, upload_every_n_epochs: int = 50):
        super().__init__()
        self.video_folder = video_folder
        self.upload_every_n_epochs = upload_every_n_epochs
        self._uploaded: set[str] = set()
        self._epoch_count = 0

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)

        if self.upload_every_n_epochs <= 0:
            return

        self._epoch_count += 1
        if self._epoch_count % self.upload_every_n_epochs != 0:
            return

        if not os.path.isdir(self.video_folder):
            return

        import wandb  # lazy — only needed when tracking

        video_files = sorted(glob.glob(os.path.join(self.video_folder, "*.mp4")))
        new_videos = [v for v in video_files if v not in self._uploaded]
        if not new_videos:
            return

        # Upload only the most recent new video to stay within storage budget.
        latest = new_videos[-1]
        try:
            wandb.log({"train/video": wandb.Video(latest, fps=30, format="mp4")}, step=frame)
            print(f"[WandbObserver] Uploaded video: {os.path.basename(latest)} (epoch {epoch_num})")
        except Exception as e:
            print(f"[WandbObserver] Video upload failed: {e}")

        # Mark all new videos as seen; delete older ones to free local disk.
        for v in new_videos[:-1]:
            try:
                os.remove(v)
            except OSError:
                pass
            self._uploaded.discard(v)
        self._uploaded.add(latest)


class RecordMosaicWrapper(gym.Wrapper):
    """Stitch 4 viewport-shifted renders into a 2×2 mosaic. No TiledCamera needed.

    For each video frame, cycles the viewport through env indices 0–3 by calling
    ``viewport_camera_controller.set_view_env_index()``, issues ``sim.render()`` to
    fire the tracking callback, then reads the annotator.  Falls back to a single-env
    render if the controller is unavailable or fewer than 4 envs exist.
    """

    NUM_RECORD_ENVS = 4

    def render(self):
        import numpy as np

        base_env = self.env.unwrapped
        controller = getattr(base_env, "viewport_camera_controller", None)
        if controller is None or base_env.num_envs < self.NUM_RECORD_ENVS:
            return self.env.render()

        original_idx = controller.cfg.env_index
        frames = []
        for env_idx in range(self.NUM_RECORD_ENVS):
            controller.set_view_env_index(env_idx)
            base_env.sim.render()                    # fires _update_tracking_callback → camera repositions
            frame = base_env.render(recompute=True)  # read annotator; skip redundant sim.render()
            if frame is not None:
                frames.append(frame)

        controller.set_view_env_index(original_idx)  # restore

        if len(frames) < self.NUM_RECORD_ENVS:
            return self.env.render()

        top    = np.concatenate([frames[0], frames[1]], axis=1)
        bottom = np.concatenate([frames[2], frames[3]], axis=1)
        return np.concatenate([top, bottom], axis=0)  # (2H, 2W, 3)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # Rollout batch size = num_envs * horizon_length; minibatch_size must divide it for rl_games PPO.
    n_env = env_cfg.scene.num_envs
    hz = int(agent_cfg["params"]["config"]["horizon_length"])
    rollout = n_env * hz
    mb = int(agent_cfg["params"]["config"]["minibatch_size"])
    if rollout % mb != 0:
        mb = rollout
        agent_cfg["params"]["config"]["minibatch_size"] = mb
        print(f"[INFO] Set minibatch_size={mb} to match num_envs*horizon_length ({n_env}*{hz}).")

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
    else:
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)

    # lower viewport resolution when recording — each tile is 256×144, mosaic output 512×288
    if args_cli.video:
        env_cfg.viewer.resolution = (256, 144)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        env = RecordMosaicWrapper(env)  # 4-env 2×2 mosaic via viewport cycling
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # initialise wandb before building the runner so AicWandbObserver can use it
    global_rank = int(os.getenv("RANK", "0"))
    use_wandb = args_cli.track and global_rank == 0
    if use_wandb:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=False,  # we handle video upload manually with AicWandbObserver
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    # create runner from rl-games
    video_folder = os.path.join(log_root_path, log_dir, "videos", "train")
    if use_wandb and args_cli.video:
        base_observer = AicWandbObserver(
            video_folder=video_folder,
            upload_every_n_epochs=args_cli.video_upload_every,
        )
    else:
        base_observer = IsaacAlgoObserver()

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([base_observer, PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(base_observer)

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

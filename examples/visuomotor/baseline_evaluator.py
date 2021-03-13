import glob
import os
import pickle
import uuid
from os.path import abspath, dirname, join

import gym
import numpy as np
import stable_baselines3
from gym import spaces
from pyrep import PyRep
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.kinova3 import Kinova3
from pyrep.robots.arms.panda import Panda
from scipy.interpolate import interp1d
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)

from baseline_door_env_training import ReacherEnv



# Hyperparams
POLICY_UPDATE_STEPS = 50
EPISODE_LENGTH = 150
TOTAL_TIMESTEPS = 20000

MODEL_PATH = "/home/lucy/Desktop/Anirudh/Master-Thesis/NJ-2020-thesis/PyRep/" \
                "examples/visuomotor/models/50_150_8000_130/96/gpu_4000_steps.zip"

env = ReacherEnv(headless=False)
env.seed(666)
env = Monitor(env)

# +++++ Evaluation +++++
model = PPO.load(MODEL_PATH \
, n_steps = EPISODE_LENGTH, n_epochs= POLICY_UPDATE_STEPS,env=env, verbose=2)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
print("====>","mean_reward : ",mean_reward ,"std_reward : ",std_reward)

env.close()
env.shutdown()
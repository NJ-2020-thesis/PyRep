import glob
import os
import pickle
import uuid
from os.path import abspath, dirname, join
import time

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

# ==================================================
# Paths to save values
log_path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/log/"

SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_kinova3_door_env_random.ttt')

# Hyperparams
POLICY_UPDATE_STEPS = 50
EPISODE_LENGTH = 300
TOTAL_TIMESTEPS = 100000

NEW_TRAINING = True

# ==================================================

model = None
for i in range(10):
    model_save_path = './models/test1/'
    print("Model save path : ",model_save_path)
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_save_path+str(i),
                                         name_prefix='norm_gpu_new')

    env = ReacherEnv(headless=True)
    env.seed(666)
    env = Monitor(env)

    if NEW_TRAINING:
        if i == 0:
            model = PPO('MlpPolicy',n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
                        env = env, verbose = 2, tensorboard_log = log_path)

        else:
            # +++++ Training +++++
            print("======>>>> Running <<<<======",i)
            model_folder = model_save_path +str(i-1) +"/*"
            list_of_files = glob.glob(model_folder) 
            latest_file = max(list_of_files, key=os.path.getctime)
            model = PPO.load(latest_file,  n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
                        env = env, verbose = 2, tensorboard_log = log_path)
    
    else:
        model_folder = model_save_path + "/*"
        latest_model_folder = max(glob.glob(model_folder), key=os.path.getctime)
        print(latest_model_folder)
        latest_model = max(glob.glob(latest_model_folder + "/*"), key=os.path.getctime)
        print(latest_model)
        # model = PPO.load(latest_model,  n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
        #                 env = env, verbose = 2, tensorboard_log = log_path)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    env.shutdown()
    time.sleep(2)
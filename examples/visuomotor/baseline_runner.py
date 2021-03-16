import glob
import os
import pickle
import uuid
import time
from os.path import abspath, dirname, join
from timeit import default_timer as timer

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
from stable_baselines3 import TD3
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
log_path = "./models/TD3/10_150_8000_130/log/"
model_save_path = './models/TD3/10_150_8000_130/'

load_path = "/home/lucy/Desktop/Anirudh/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/models/50_150_8000_130_(3)/225/gpu_8000_steps.zip"

SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_kinova3_door_env_random.ttt')

# Hyperparams
POLICY_UPDATE_STEPS = 100
EPISODE_LENGTH = 500
TOTAL_TIMESTEPS = 50000
TOTAL_LOOPS = 5

CURRENT_CHECKPNT = 225
NEW_TRAINING = True
# ==================================================

start_full = timer()


model = None
for i in range(TOTAL_LOOPS):
    print("Model save path : ",model_save_path)
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=model_save_path+str(CURRENT_CHECKPNT+ i),
                                         name_prefix='gpu')

    env = ReacherEnv(headless=False)
    env.seed(666)
    env = Monitor(env)

    if NEW_TRAINING:
        if i == 0:
            model = PPO('MlpPolicy',n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
                        env = env, verbose = 2, tensorboard_log = log_path)
            print(model.policy.optimizer)
            # model = PPO.load(load_path,  n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
            #             env = env, verbose = 2, tensorboard_log = log_path)
        else:
            # +++++ Training +++++
            print("======>>>> Running <<<<======",i)
            model_folder = model_save_path + str(CURRENT_CHECKPNT + i - 1) +"/*"
            list_of_files = glob.glob(model_folder)
            # print(list_of_files)
            latest_file = max(list_of_files, key=os.path.getctime)
            model = PPO.load(latest_file,  n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
                        env = env, verbose = 2, tensorboard_log = log_path)
            print(model.policy.optimizer)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    print(model.policy.optimizer)

    del model
    env.shutdown()
    time.sleep(2)

end_full = timer()
print(" ")
print("Final Execution Time ==> ",end_full - start_full)

import glob
import os
import gym
import numpy as np
import stable_baselines3
from gym import spaces
import pickle
import time
import uuid
from os.path import abspath, dirname, join
from timeit import default_timer as timer
from scipy.interpolate import interp1d


from pyrep import PyRep
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.kinova3 import Kinova3
from pyrep.robots.arms.panda import Panda

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from baseline_door_env_training import ReacherEnv
# ==================================================
# Paths to save values
log_path = "./models/mp_trial_1/log/"
model_save_path = './models/mp_trial_1/'

SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_kinova3_door_env_random.ttt')

# Hyperparams
POLICY_UPDATE_STEPS = 20
EPISODE_LENGTH = 150
TOTAL_TIMESTEPS = 8000

NEW_TRAINING = True
# ==================================================

def create_env():
    env = ReacherEnv(headless=True)
    # env.seed(666)
    env = Monitor(env)

    return env

start_full = timer()

model = None
for i in range(30):
    print("Model save path : ",model_save_path)
    checkpoint_callback = CheckpointCallback(save_freq=4000, save_path=model_save_path+str(i),
                                         name_prefix='gpu')

    env = SubprocVecEnv([create_env() for i in range(2)])

    if NEW_TRAINING:
        if i == 0:
            model = PPO('MlpPolicy',n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
                        env = env, verbose = 2, tensorboard_log = log_path)

        else:
            # +++++ Training +++++
            print("======>>>> Running <<<<======",i)
            model_folder = model_save_path + str(i-1) +"/*"
            list_of_files = glob.glob(model_folder) 
            print(list_of_files)
            latest_file = max(list_of_files, key=os.path.getctime)
            model = PPO.load(latest_file,  n_steps = EPISODE_LENGTH, n_epochs = POLICY_UPDATE_STEPS,
                        env = env, verbose = 2, tensorboard_log = log_path)
    

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    env.shutdown()
    time.sleep(2)

end_full = timer()
print(" ")
print("Final Execution Time ==> ",end_full - start_full)

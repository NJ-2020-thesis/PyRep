# https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html?highlight=output
# http://amid.fish/reproducing-deep-rl?fbclid=IwAR1VPZm3FSTrV8BZ4UdFc2ExZy0olusmaewmloTPhpA4QOnHKRI2LLOz3mM
# https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html

import pickle
import uuid
from os.path import abspath, dirname, join
from scipy.interpolate import interp1d

import gym
import numpy as np
from gym import spaces
from pyrep import PyRep
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.kinova3 import Kinova3
from pyrep.robots.arms.panda import Panda
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# Paths to save values
rewards_file = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/results/rewards"+ str(uuid.uuid4())[:8] +"_1000.txt"
model_path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/results/model_"+ str(uuid.uuid4())[:8] +"_1000.mdl"
log_path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/log/"

SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_kinova3_door_env_random.ttt')

# Hyperparams
POLICY_UPDATE_STEPS = 50
EPISODE_LENGTH = 200
TOTAL_TIMESTEPS = 100000

# GPU
print("GPU Available : ",stable_baselines3.common.utils.get_device())

class ReacherEnv(gym.Env):

    def __init__(self):
        super(ReacherEnv,self).__init__()

        # Creating a Custom Pyrep environment 
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()

        # SavinCreating the Robot and Other objects
        # These will be used for collision, action space etc.
        self.agent = Kinova3()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.vision_sensor = VisionSensor("vision_sensor")
        self.vision_sensor.set_resolution([64,64])

        self.dining_table = Shape('diningTable')

        self.handle = Shape("door_handle_visible")
        handle_bounding_box = self.handle.get_bounding_box()

        # Saving Env initial state
        self.gripper = Shape('ROBOTIQ_85')
        self.gripper_state = self.gripper.get_configuration_tree()
        self.door = Shape('door_frame')
        self.door_state = self.door.get_configuration_tree()


        # Bounding box within wich we get successful grasp
        BOUNDING_BOX_EPS = 0.003
        self.target = Dummy('start_point3')
        position_min, position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*BOUNDING_BOX_EPS]
        self.random_handle_pos = list(np.random.uniform(position_min, position_max))
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)

        # Setting action and state space for robot 
        self.observation_space = spaces.Box(low=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]),
                                     high=np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]]), dtype=np.float)

        # Normalize between [-1,1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        # self.action_space = spaces.Box(low=np.asarray([-val[0] for val in self.agent.get_joint_intervals()[1]]),
        #                              high=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]), dtype="float32")
        # self.action_space = spaces.Box(low=np.asarray([-val for val in self.agent.get_joint_upper_velocity_limits()]),
        #                              high=np.asarray([val for val in self.agent.get_joint_upper_velocity_limits()]), dtype=np.float)


    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # initial_image = self.vision_sensor.capture_rgb()
        # print(self.agent.get_joint_intervals())
        # initial_representation = 
        # goal_representation = 
        return np.concatenate([self.agent.get_joint_positions()])

    def reset(self):
        # Reset environment with initial conditions
        self.agent.set_joint_target_positions(self.initial_joint_positions)
        self.pr.set_configuration_tree(self.gripper_state)
        self.pr.set_configuration_tree(self.door_state)
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)
        return self._get_state()

    def step(self, action):
        done = False

        # Denorm values
        lower_limits = [val[0] for val in self.agent.get_joint_intervals()[1]]
        upper_limits = [val[1] for val in self.agent.get_joint_intervals()[1]]
        denorm_action = [] 

        for num,low,high in zip(action,lower_limits,upper_limits):
            new = np.interp(num,[-1,1],[low,high])
            denorm_action.append(new)

        # print(lower_limits)
        # print(upper_limits)
        # print(action)
        # print(denorm_action)
        # print("")
        # self.agent.set_joint_target_velocities(action)
        self.agent.set_joint_target_positions(denorm_action)

        self.pr.step()  # Step the physics simulation

        # Reward is negative distance to target
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()

        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        info = {}

        # Simple collision detection
        if self.dining_table.check_collision() == True :
            print(self.dining_table.check_collision())
            done = True
            return self._get_state(),reward,done,info

        return self._get_state(),reward,done,info

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

# ------------------------------------------------------------------
def make_env(env_id, rank, seed=666):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ReacherEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                         name_prefix='rl_model_norm_gpu')

# env = SubprocVecEnv([make_vec_env("abc", i) for i in range(4)])
env = ReacherEnv()
env.seed(666)
model = PPO('MlpPolicy',n_steps = EPISODE_LENGTH, n_epochs= POLICY_UPDATE_STEPS ,
                env=env, verbose=2, tensorboard_log=log_path)

print(env.spec)

# +++++ Training +++++

# model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
# model.save(model_path)

# +++++ Evaluation +++++
# model.load("/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/logs/rl_model_400000_steps.zip")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
# print("====>","mean_reward : ",mean_reward ,"std_reward : ",std_reward)


# +++++ Random Eval +++++

# env = ReacherEnv()
# obs = env.reset()
# n_steps = 10000
# for i in range(n_steps):
#     # Random action
#     action = env.action_space.sample()
#     # print(action)
#     obs, reward, done, info = env.step(action)
#     if done or (i%3000 == 0):
#         obs = env.reset()

env.close()
env.shutdown()

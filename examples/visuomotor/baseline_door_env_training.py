# https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html?highlight=output
# http://amid.fish/reproducing-deep-rl?fbclid=IwAR1VPZm3FSTrV8BZ4UdFc2ExZy0olusmaewmloTPhpA4QOnHKRI2LLOz3mM
# https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html
# https://medium.com/pytorch/robotic-assembly-using-deep-reinforcement-learning-dfd9916c5ad7

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
from pyrep.objects.proximity_sensor import ProximitySensor 

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

        self.proximity_sensor = ProximitySensor('ROBOTIQ_85_attachProxSensor')

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

        self.initial_distance = self.distance_to_goal()
        # print("Initial distance : ",self.initial_distance)
        # Normalize between [-1,1]
        self.action_space = spaces.Box(low=-1., high=1., shape=(7,), dtype=np.float32)
        # self.action_space = spaces.Box(low=np.asarray([-val[0] for val in self.agent.get_joint_intervals()[1]]),
        #                              high=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]), dtype="float32")
        # self.action_space = spaces.Box(low=np.asarray([-val for val in self.agent.get_joint_upper_velocity_limits()]),
        #                              high=np.asarray([val for val in self.agent.get_joint_upper_velocity_limits()]), dtype=np.float)

    def distance_to_goal(self):
        # Reward is negative distance to target
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()

        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        return reward

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # initial_image = self.vision_sensor.capture_rgb()
        # initial_representation = 
        # goal_representation = 
        joint_pos = self.agent.get_joint_positions()
        return np.concatenate([joint_pos])

    def reset(self):
        # Reset environment with initial conditions
        self.agent.set_joint_target_positions(self.initial_joint_positions)
        self.pr.set_configuration_tree(self.gripper_state)
        self.pr.set_configuration_tree(self.door_state)
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)
        self.initial_distance = self.distance_to_goal()

        return self._get_state()

    def step(self, action):
        done = False
        info = {}
        prev_distance_to_goal = self.distance_to_goal()

        # Denorm values
        lower_limits = [val[0] for val in self.agent.get_joint_intervals()[1]]
        upper_limits = [val[1] for val in self.agent.get_joint_intervals()[1]]
        denorm_action = [] 

        for num,low,high in zip(action,lower_limits,upper_limits):
            new = np.interp(num,[-1,1],[low,high])
            denorm_action.append(new)

        # self.agent.set_joint_target_velocities(action)
        self.agent.set_joint_target_positions(denorm_action)
        self.pr.step()  # Step the physics simulation

        # Reward calculations
        success_reward = self.success_check()
        distance_reward = (prev_distance_to_goal - self.distance_to_goal()) / self.initial_distance
        reward = distance_reward + success_reward

        # Simple collision detection
        if self.dining_table.check_collision() == True :
            done = True
            return self._get_state(),reward,done,info
        elif self.distance_to_goal() > 0.8:
            done = True
        
        print(reward)
        return self._get_state(),reward,done,info

    def success_check(self):
        success_reward = 0.
        if self.distance_to_goal() < 0.01:
            success_reward = 10
        return success_reward

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

## ------------------------------------------------------------------

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/1/',
                                         name_prefix='norm_gpu_new')

env = ReacherEnv()
env.seed(666)
model = PPO('MlpPolicy',n_steps = EPISODE_LENGTH, n_epochs= POLICY_UPDATE_STEPS ,
                env=env, verbose=2, tensorboard_log=log_path)

print(env.spec)

## +++++ Training +++++
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save(model_path)

# # +++++ Evaluation +++++
# model.load("/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/models/norm_gpu_new_800000_steps.zip")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
# print("====>","mean_reward : ",mean_reward ,"std_reward : ",std_reward)


## +++++ Random Eval +++++

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

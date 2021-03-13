# https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html?highlight=output
# http://amid.fish/reproducing-deep-rl?fbclid=IwAR1VPZm3FSTrV8BZ4UdFc2ExZy0olusmaewmloTPhpA4QOnHKRI2LLOz3mM
# https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2
# https://stable-baselines.readthedocs.io/en/master/guide/examples.html
# https://medium.com/pytorch/robotic-assembly-using-deep-reinforcement-learning-dfd9916c5ad7

import pickle
import random
import uuid
import cv2
import glob
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
from pyrep.textures.texture import Texture
from pyrep.const import TextureMappingMode, RenderMode

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# Texture paths
DOOR_TEXTURES = glob.glob("/home/anirudh/HBRS/Master-Thesis/Images/Textures/Waste/*.jpg")
HANDLE_TEXTURES = glob.glob("/home/anirudh/HBRS/Master-Thesis/Images/Textures/Handle/*.jpg")


SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_kinova3_door_env_random.ttt')

# Hyperparams
POLICY_UPDATE_STEPS = 25
EPISODE_LENGTH = 100
TOTAL_TIMESTEPS = 100000

counter = 0

# GPU
print("GPU Available : ",stable_baselines3.common.utils.get_device())

class ReacherEnv(gym.Env):

    def __init__(self,headless=True):
        super(ReacherEnv,self).__init__()

        # Creating a Custom Pyrep environment
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()

        # -------------------------------------------------
        # Creating the Robot and Other objects
        # These will be used for collision, action space etc.
        self.agent = Kinova3()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.agent_state = self.agent.get_configuration_tree()

        self.vision_sensor = VisionSensor("vision_sensor")
        self.vision_sensor.set_render_mode(render_mode=RenderMode.OPENGL3)
        self.vision_sensor.set_resolution([128,128])

        self.dining_table = Shape('diningTable')

        self.handle = Shape("door_handle_visible")
        handle_bounding_box = self.handle.get_bounding_box()

        self.proximity_sensor = ProximitySensor('ROBOTIQ_85_attachProxSensor')

        self.step_counter = 0

        # -------------------------------------------------
        # Saving Env initial state
        self.gripper = Shape('ROBOTIQ_85')
        self.gripper_state = self.gripper.get_configuration_tree()

        self.door = Shape('door_frame')
        self.door_state = self.door.get_configuration_tree()
        self.door_surface = Shape('door_main_visible')

        # -------------------------------------------------
        # Bounding box within wich we get successful grasp
        BOUNDING_BOX_EPS = 0.003
        self.target = Dummy('start_point3')
        self.position_min, self.position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*BOUNDING_BOX_EPS]
        self.random_handle_pos = list(np.random.uniform(self.position_min, self.position_max))
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)
        self.target.set_orientation([np.deg2rad(0), \
                                    np.deg2rad(-90),\
                                    np.deg2rad(0)])

        # -------------------------------------------------
        # Setting action and state space for robot
        self.observation_space = spaces.Box(low=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]),
                                     high=np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]]), dtype=np.float)

        self.initial_distance = self.reward_distance_to_goal()

        # Normalize between [-1,1]
        self.action_space = spaces.Box(low=-1., high=1., shape=(7,), dtype=np.float32)
        # self.action_space = spaces.Box(low=np.asarray([-val[0] for val in self.agent.get_joint_intervals()[1]]),
        #                              high=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]), dtype="float32")
        # self.action_space = spaces.Box(low=np.asarray([-val for val in self.agent.get_joint_upper_velocity_limits()]),
        #                              high=np.asarray([val for val in self.agent.get_joint_upper_velocity_limits()]), dtype=np.float)
        self.pr.step()

    def reset(self):

        self.setup_scene()
        print(self.initial_joint_positions, self.agent.get_joint_positions())
        return self._get_state()

    def step(self, action):
        done = False
        info = {}
        prev_distance_to_goal = self.reward_distance_to_goal()

        # Denorm values
        lower_limits = [-val for val in self.agent.get_joint_upper_velocity_limits()]
        upper_limits = [val for val in self.agent.get_joint_upper_velocity_limits()]

        denorm_action = []

        for num,low,high in zip(action,lower_limits,upper_limits):
            new = np.interp(num,[-1,1],[low,high])
            denorm_action.append(new)

        self.agent.set_joint_target_velocities(denorm_action) # Try this
        # self.agent.set_joint_target_positions(denorm_action)
        self.pr.step()  # Step the physics simulation

        # Reward calculations
        success_reward, success = self.reward_success()
        angle_reward = self.reward_orientation()
        # distance_reward = self.reward_distance_to_goal()
        distance_reward = (prev_distance_to_goal - self.reward_distance_to_goal())/self.initial_distance # Relative reward
        reward = (distance_reward * 10) + success_reward + angle_reward

        #------------------------------------------------
        if self.step_counter % EPISODE_LENGTH == 0:
            done = True
            print('--------Reset: Timeout--------')

        if success:
            done = True
            print('--------Reset: Success is true--------')

        if self.dining_table.check_collision() or self.door.check_collision(obj=self.agent):
            done = True
            reward += -100
            print("----- Reset: Collision -----")
            return self._get_state(),reward,done,info

        self.step_counter += 1
        print(self.step_counter," ",distance_reward*10," ",success_reward," ",angle_reward," ",reward)
        return self._get_state(),reward,done,info

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        global counter

        initial_image = self.vision_sensor.capture_rgb()
        # cv2.imwrite("/home/anirudh/Desktop/result/"+str(counter)+".jpg",initial_image*255)
        counter += 1

        # initial_representation =
        # goal_representation =
        joint_pos = self.agent.get_joint_positions()
        return np.concatenate([joint_pos])

    # ---------------------------REWARDS-----------------------------
    def reward_distance_to_goal(self):
        # Reward is negative distance to target
        agent_position = self.agent_ee_tip.get_position()
        target_position = self.target.get_position()

        # reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        reward = np.linalg.norm(agent_position - target_position)

        # Reset environment with initial conditions
        self.agent.set_joint_target_velocities([0,0,0,0,0,0,0])
        self.agent.set_joint_target_positions([0,0,0,0,0,0,0])

        return reward

    def reward_orientation(self):
        agent_orientation = self.agent_ee_tip.get_orientation()
        target_orientation = self.target.get_orientation()

        orientation_value =  (np.dot( agent_orientation, target_orientation )
        / max( np.linalg.norm(agent_orientation) * np.linalg.norm(target_orientation), 1e-10 ))

        return abs(orientation_value)


    def reward_success(self):
        DISTANCE = 0.03
        success_reward = -1 # default reward per timestep
        success = False

        # print(self.proximity_sensor.is_detected(self.handle)," ",-1 != self.proximity_sensor.read() < DISTANCE," ",self.proximity_sensor.read())
        if self.proximity_sensor.read() < DISTANCE and \
                self.proximity_sensor.read() != -1 and \
                self.proximity_sensor.is_detected(self.handle):
            success_reward = +100000.0
            success = True

        # if self.proximity_sensor.read() < DISTANCE and \
        #         self.proximity_sensor.read() != -1:
        #     success_reward = +100000.0
        #     success = True

        return success_reward, success

    # ------------------------------RESET-------------------------
    def setup_scene(self):
        # ----------------------------------------------
        # ROBOT POSE RANDOMIZATION
        random_pose = list(np.arange(-0.05,0.05,0.002))
        eps = random.sample(random_pose, 7)
        eps[6] = random.sample(list(np.arange(-3.0,3.0,0.2)),1)[0] # randomizing the gripper
        random_start_joint_positions = np.add(self.initial_joint_positions ,eps)
        self.agent.set_joint_positions(random_start_joint_positions)

        # ----------------------------------------------
        # ENV RANDOMIZATION
        # self.pr.set_configuration_tree(self.gripper_state)
        # self.pr.set_configuration_tree(self.door_state)

        # # ---> Door
        # door_path = random.choice(DOOR_TEXTURES)
        # d_text_ob, door_texture = self.pr.create_texture(door_path)
        # for item in self.door_surface.ungroup():
        #     item.remove_texture()
        #     if item.get_name() == "Plane_4":
        #         # item.remove_texture()
        #         item.set_texture(door_texture,TextureMappingMode.PLANE)
        # d_text_ob.remove()

        # # ---> Handle
        # h_text_ob, handle_texture = self.pr.create_texture(random.choice(HANDLE_TEXTURES))
        # self.handle.remove_texture()
        # self.handle.set_texture(handle_texture,TextureMappingMode.CYLINDER)
        # h_text_ob.remove()

        # ----------------------------------------------
        # TARGET RANDOMIZATION
        self.random_handle_pos = list(np.random.uniform(self.position_min, self.position_max))
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

## ------------------------------------------------------------------

if __name__ == "__main__":
    pass

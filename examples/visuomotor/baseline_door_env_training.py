import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from pyrep.objects.vision_sensor import VisionSensor 

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.kinova3 import Kinova3
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
import numpy as np
import pickle

rewards_file = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/results/rewards_1000.txt"
model_path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/results/model_saved_1000.mdl"
log_path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/log/"

SCENE_FILE = join(dirname(abspath(__file__)),
                  'scene_kinova3_door_env_random.ttt')

POLICY_UPDATE_STEPS = 50
EPISODE_LENGTH = 300
TOTAL_TIMESTEPS = 1000

class ReacherEnv(gym.Env):

    def __init__(self):
        super(ReacherEnv,self).__init__()

        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()

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

        # Env initial state
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
        # self.action_space = spaces.Box(low=np.asarray([val[0] for val in self.agent.get_joint_intervals()[1]]),
        #                              high=np.asarray([val[1] for val in self.agent.get_joint_intervals()[1]]), dtype=np.float)
        self.action_space = spaces.Box(low=np.asarray([-val for val in self.agent.get_joint_upper_velocity_limits()]),
                                     high=np.asarray([val for val in self.agent.get_joint_upper_velocity_limits()]), dtype=np.float)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(7,), dtype=np.float)


    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        # print(self.agent.get_joint_intervals())
        # print(self.agent.get_joint_upper_velocity_limits())
        initial_image = self.vision_sensor.capture_rgb()
        # initial_representation = 
        # goal_representation = 
        return np.concatenate([self.agent.get_joint_positions()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        # Reset env
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.pr.set_configuration_tree(self.gripper_state)
        self.pr.set_configuration_tree(self.door_state)
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)
        return self._get_state()

    def step(self, action):
        if self.dining_table.check_collision() == True :
            self.reset()

        self.agent.set_joint_target_velocities(action)
        print(self.dining_table.check_collision())

        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()

        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        info = {}
        done = False
        # print(reward)

        return self._get_state(),reward,done,info

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

load_model_path = "/home/anirudh/HBRS/Master-Thesis/NJ-2020-thesis/PyRep/examples/visuomotor/results/ppo_saved_test.mdl"
env = ReacherEnv()
model = PPO('MlpPolicy',n_steps = EPISODE_LENGTH, n_epochs= POLICY_UPDATE_STEPS ,env=env, verbose=2,tensorboard_log=log_path)
# model.learn(total_timesteps=TOTAL_TIMESTEPS)
# model.save(model_path)

model.load(load_model_path,env)

# check_env(env, warn=True)


# rewards = []
# obs = env.reset()
# for i in range(EPISODE_LENGTH):

#     action, _states = model.predict(obs, deterministic=True)
#     obs,reward,done,info = env.step(action)
#     rewards.append(reward)
#     if done:
#       obs = env.reset()
#     print("ENV : ",i," ", "REWARD : ",reward)

# np.save(rewards_file,rewards,allow_pickle=False)

env.close()
env.shutdown()
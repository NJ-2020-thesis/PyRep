import gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
EPISODES = 100000
EPISODE_LENGTH = 100000


class ReacherEnv(gym.Env):

    def __init__(self):
        super(ReacherEnv,self).__init__()

        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()

        self.agent = Kinova3()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)

        self.target = Dummy('start_point3')
        print(self.target.get_position())

        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.handle = Shape("door_handle_visible")
        handle_bounding_box = self.handle.get_bounding_box()

        # Bounding box within wich we get successful grasp
        BOUNDING_BOX_EPS = 0.003
        position_min, position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*BOUNDING_BOX_EPS]
        self.random_handle_pos = list(np.random.uniform(position_min, position_max))
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)
        print(self.target.get_position())

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(7,), dtype=np.float)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(17,), dtype=np.float)


    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.target.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        self.target.set_position(position = self.random_handle_pos,relative_to=self.handle)
        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        info = {}
        done = False
        print(reward)
        # print("--->",self._get_state(),self._get_state().shape)

        return self._get_state(),reward,done,info

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


env = ReacherEnv()
model = PPO('MlpPolicy', env, verbose=2,tensorboard_log=log_path)
model.learn(total_timesteps=EPISODE_LENGTH)
model.save(model_path)

# check_env(env, warn=True)

rewards = []

obs = env.reset()
for i in range(EPISODES):

    action, _states = model.predict(obs, deterministic=True)
    obs,reward,done,info = env.step(action)
    rewards.append(reward)
    if done:
      obs = env.reset()
    print("ENV : ",i," ", "REWARD : ",reward)

np.save(rewards_file,rewards,allow_pickle=False)

env.close()
env.shutdown()
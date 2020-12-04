from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.robots.arms.kinova3 import Kinova3
from pyrep.objects.camera import Camera
from pyrep.objects.vision_sensor import VisionSensor 
from pyrep.objects.proximity_sensor import ProximitySensor 

from utils.dataset_generator import *
from utils.manip_utils import move_arm,move_arm_to_end

import numpy as np
import math
import time
import random
from os import path


EPISODES = 10
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_kinova3_door_env_random.ttt')

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.set_simulation_timestep(0.2)
pr.start()


print("------------LOGIC-------------")

agent = Kinova3()
agent.set_control_loop_enabled(True)
agent.set_motor_locked_at_zero_velocity(True)
starting_joint_positions = agent.get_joint_positions()
agent.set_joint_positions(starting_joint_positions)

gripper = Shape('ROBOTIQ_85')
gripper_state = gripper.get_configuration_tree()

handle = Shape("door_handle_visible")
handle_bounding_box = handle.get_bounding_box()
handle_fixed_position = handle.get_position()

door = Shape('door_frame')
door_state = door.get_configuration_tree()

proximity_sensor = ProximitySensor('ROBOTIQ_85_attachProxSensor')
# ---------------------------------------

target = Dummy('start_point3')

# Bounding box within wich we get successful grasp
BOUNDING_BOX_EPS = 0.01
position_min, position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*BOUNDING_BOX_EPS]

# Random pose for robot 
random_pose = list(np.arange(-0.03,0.03,0.002))

for i in range(EPISODES):

    eps = random.sample(random_pose, 7)
    eps[6] = random.sample(list(np.arange(-3.0,3.0,0.4)),1)[0]
    agent.set_joint_positions(np.add(starting_joint_positions,eps))
    # agent.set_joint_positions([0,0,0,0,0,0,0])

    pr.set_configuration_tree(gripper_state)
    pr.set_configuration_tree(door_state)

    target.set_position(position = list(np.random.uniform(position_min, position_max)),relative_to=handle)
        # target.set_position([0,0,0],relative_to=handle)
    print()
    try :
        move_arm_to_end(pr,agent,proximity_sensor,target.get_position(),target.get_quaternion(),target.get_orientation(),True)
        print(handle_fixed_position)
        print(handle.get_position())
        print("Successful! ",i)
    except ConfigurationPathError as e:
        print("SKIPPING")
        continue


print("--------------------------------")

pr.stop()
pr.shutdown()


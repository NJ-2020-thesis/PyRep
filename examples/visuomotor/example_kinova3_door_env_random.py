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

from utils.dataset_generator import *
from utils.manip_utils import move_arm

import numpy as np
import math
import time
import random
from os import path


EPISODES = 2
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_kinova3_door_env_random.ttt')

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
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

door = Shape('door_frame')
door_state = door.get_configuration_tree()

# ---------------------------------------

target = Dummy('start_point3')

position_min, position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*0.01]
random_pose = list(np.arange(-0.05,0.05,0.002))

for i in range(EPISODES):

    eps = random.sample(random_pose, 7)
    eps[6] = random.sample(list(np.arange(-3.0,3.0,0.4)),1)[0]
    agent.set_joint_positions(np.add(starting_joint_positions,eps))
    # agent.set_joint_positions([0,0,0,0,0,0,0])
    # print(agent.get_joint_intervals())

    pr.set_configuration_tree(gripper_state)
    pr.set_configuration_tree(door_state)

    target.set_position(position = list(np.random.uniform(position_min, position_max)),relative_to=handle)
        # target.set_position([0,0,0],relative_to=handle)

    try :
    # move_arm(start_point2.get_position(),start_point2.get_quaternion(),start_point2.get_orientation(),False)
    # move_arm(start_point1.get_position(),start_point1.get_quaternion(),start_point1.get_orientation(),False)
    # move_arm(start_point0.get_position(),start_point0.get_quaternion(),start_point0.get_orientation(),False)
    # move_arm(start_point.get_position(),start_point.get_quaternion(),start_point.get_orientation(),False)
        move_arm(pr,agent,target.get_position(),target.get_quaternion(),target.get_orientation(),True)
        print("Successful!")
    except:
        print("SKIPPING")


print("--------------------------------")

pr.stop()
pr.shutdown()


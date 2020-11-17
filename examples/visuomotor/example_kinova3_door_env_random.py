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

import numpy as np
import math
import time
import random
from os import path


EPISODES = 26
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_kinova3_door_env_random.ttt')

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()


print("------------LOGIC-------------")

agent = Kinova3()
agent.set_control_loop_enabled(True)
agent.set_motor_locked_at_zero_velocity(True)
initial_joint_positions = agent.get_joint_positions()
agent.set_joint_positions(initial_joint_positions)

handle = Shape("door_handle_visible")
handle.set_color([0.5,0.5,1.0])
handle_bounding_box = handle.get_bounding_box()

starting_joint_positions = agent.get_joint_positions()

def move_arm(position, quaternion, orientation, ignore_collisions=False):
    arm_path = agent.get_path(position=position,
                            # euler=[0, math.radians(180), 0],
                            euler=[0, 0, 0],
                            ignore_collisions=ignore_collisions)
    arm_path.visualize()
    done = False

    while not done:
        done = arm_path.step()
        pr.step()

    arm_path.clear_visualization()



start_point = Dummy('start_point')
start_point0 = Dummy('start_point0')
start_point1 = Dummy('start_point1')
start_point2 = Dummy('start_point2')

target = Dummy('start_point3')

position_min, position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*0.3]

for i in range(EPISODES):

    eps = random.sample(list(np.arange(0.01,0.03,0.001)), 7)
    agent.set_joint_positions(np.add(starting_joint_positions,eps))

        # print(position_min,position_max)
    target.set_position(position = list(np.random.uniform(position_min, position_max)),relative_to=handle)
        # target.set_position([0,0,0],relative_to=handle)

    try :
    # move_arm(start_point2.get_position(),start_point2.get_quaternion(),start_point2.get_orientation(),False)
    # move_arm(start_point1.get_position(),start_point1.get_quaternion(),start_point1.get_orientation(),False)
    # move_arm(start_point0.get_position(),start_point0.get_quaternion(),start_point0.get_orientation(),False)
    # move_arm(start_point.get_position(),start_point.get_quaternion(),start_point.get_orientation(),False)
        move_arm(target.get_position(),target.get_quaternion(),target.get_orientation(),True)

    except:
        print("SKIPPING")


print("--------------------------------")

pr.stop()
pr.shutdown()



# [-0.07000000029802322, -0.047999996691942215, -0.03500000014901161] 
# [0.07000000029802322, 0.047999996691942215, 0.03500000014901161]

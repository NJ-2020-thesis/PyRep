"""
A Franka Panda reaches for 10 randomly places targets.
This script contains examples of:
    - Linear (IK) paths.
    - Scene manipulation (creating an object and moving it).
"""
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

EPISODES = 20
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_kinova3_door_env.ttt')

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

DATASET_PATH = "/home/anirudh/Desktop/Dataset/"

print("------------LOGIC-------------")

agent = Kinova3()
agent.set_control_loop_enabled(True)
agent.set_motor_locked_at_zero_velocity(True)
initial_joint_positions = agent.get_joint_positions()
agent.set_joint_positions(initial_joint_positions)

handle_box = Shape("handle_boundingbox")
handle_bounding_box = handle_box.get_bounding_box()

print("Agent Joints",agent.get_joint_positions())
print("Agent Pose",agent.get_pose())
print("BBox : ",handle_box.get_bounding_box())
# print("Limits : ",agent.set_joint_positions([1.,0.,0.,0.,0.,0.,0.]))
# print(" ->> : ",agent.get_joint_intervals())

starting_joint_positions = agent.get_joint_positions()

vision_sensor = VisionSensor("vision_sensor")
depth_sensor_left =  VisionSensor("depth_sensor_left")
depth_sensor_right =  VisionSensor("depth_sensor_right")

def move_arm(position, quaternion, orientation , save_path, ignore_collisions=False, save_data=False):
    arm_path = agent.get_path(position=position,
                            euler=[0, math.radians(180), 0],
                            ignore_collisions=ignore_collisions)
    arm_path.visualize()
    i,max_step = 0,1000
    done = False


    vision_data,depth_data_left,depth_data_right = [],[],[]
    joint_pose_list = []
    while not done:
        done = arm_path.step()
        pr.step()
        
    #     if save_data:
    #         vision_data.append(vision_sensor.capture_rgb())
    #         depth_data_left.append(depth_sensor_left.capture_rgb())
    #         depth_data_right.append(depth_sensor_right.capture_rgb())
    #         joint_pose_list.append(agent.get_joint_positions())

    #         print(len(vision_data),len(depth_data_left),len(vision_data),len(joint_pose_list))
        
    #     i += 1

    # if save_data: 
    #     np.save(path.join(save_path,"vision.npx"),np.array(vision_data))
    #     np.save(path.join(save_path,"depth_data_left.npx"),np.array(depth_data_left))
    #     np.save(path.join(save_path,"depth_data_right.npx"),np.array(depth_data_right))
    #     np.save(path.join(save_path,"joint_pose_list.npx"),np.array(joint_pose_list))

    arm_path.clear_visualization()


start_point = Dummy('start_point')
start_point0 = Dummy('start_point0')
start_point1 = Dummy('start_point1')
start_point2 = Dummy('start_point2')
# target = Dummy.create(0.02)

target = Shape.create(type=PrimitiveShape.SPHERE,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)


for i in range(EPISODES):

    # eps = random.sample(list(np.arange(0.01,0.3,0.001)), 7)
    # agent.set_joint_positions(starting_joint_positions + eps)

    path_dict = create_folder_structure(DATASET_PATH,i)

    # position_min, position_max = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
    position_min, position_max = [handle_bounding_box[0],handle_bounding_box[1],handle_bounding_box[2]], \
                                 [handle_bounding_box[3],handle_bounding_box[4],handle_bounding_box[5]]

    target.set_position(position = list(np.random.uniform(position_min, position_max)))
    target.set_orientation(start_point.get_orientation())

    # move_arm(start_point2.get_position(),start_point2.get_quaternion(),start_point2.get_orientation(),path_dict["episode_path"],True,False)
    # move_arm(start_point1.get_position(),start_point1.get_quaternion(),start_point1.get_orientation(),path_dict["episode_path"],True,False)
    # move_arm(start_point0.get_position(),start_point0.get_quaternion(),start_point0.get_orientation(),path_dict["episode_path"],True,False)
    move_arm(start_point.get_position(),start_point.get_quaternion(),start_point.get_orientation(),path_dict["episode_path"],True,False)
    # move_arm(target.get_position(),target.get_quaternion(),target.get_orientation(),path_dict["episode_path"],True,False)

print("--------------------------------")

pr.stop()
pr.shutdown()

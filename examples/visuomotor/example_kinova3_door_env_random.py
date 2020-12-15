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
from pyrep.textures.texture import Texture
from pyrep.const import TextureMappingMode

from utils.dataset_generator import *
from utils.manip_utils import move_arm,move_arm_to_end

import math
import time
import cv2  
import random
from os import path 
import numpy as np
import pickle
import glob

text_file = open("/home/anirudh/Desktop/Dataset/new.txt", "wb")

# ----------Variables-----------
EPISODES = 10
DATASET_PATH = "/home/anirudh/Desktop/Dataset"
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_kinova3_door_env_random.ttt')

DOOR_TEXTURES = glob.glob("/home/anirudh/HBRS/Master-Thesis/Images/Textures/*.jpg")
HANDLE_TEXTURES = glob.glob("/home/anirudh/HBRS/Master-Thesis/Images/Textures/Handle/*.jpg")

# ------------INIT-------------
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
# pr.set_simulation_timestep(0.2)
pr.start()

# print("------------LOGIC-------------")

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
handle_fixed_orientation = handle.get_orientation()

door = Shape('door_frame')
door_state = door.get_configuration_tree()
door_surface = Shape('door_main_visible')

proximity_sensor = ProximitySensor('ROBOTIQ_85_attachProxSensor')

vision_sensor = VisionSensor("vision_sensor")
vision_sensor.set_resolution([128,128])
depth_sensor_left =  VisionSensor("depth_sensor_left")
depth_sensor_right =  VisionSensor("depth_sensor_right")

target = Dummy('start_point3')
# ---------------------------------------


# Bounding box within wich we get successful grasp
BOUNDING_BOX_EPS = 0.01
position_min, position_max = [handle_bounding_box[0],handle_bounding_box[2],handle_bounding_box[4]], \
                            [handle_bounding_box[1],handle_bounding_box[3],handle_bounding_box[5]*BOUNDING_BOX_EPS]

# Random pose for robot 
random_pose = list(np.arange(-0.03,0.03,0.002))

result_list = []
for i in range(EPISODES):

    # Randomizing the initial poses of the 
    # robot and handle in the environment.
    # ROBOT POSE RANDOMIZATION
    eps = random.sample(random_pose, 7)
    eps[6] = random.sample(list(np.arange(-3.0,3.0,0.2)),1)[0] # randomizing the gripper
    random_start_joint_positions = np.add(starting_joint_positions,eps)
    agent.set_joint_positions(random_start_joint_positions)

    # Reset env
    pr.set_configuration_tree(gripper_state)
    pr.set_configuration_tree(door_state)

    # TEXTURE RANDOMIZATION
    # ---> Door
    door_path = random.choice(DOOR_TEXTURES)
    print(door_path)
    d_text_ob, door_texture = pr.create_texture(door_path)
    for item in door_surface.ungroup():
        item.remove_texture()
        item.set_texture(door_texture,TextureMappingMode.SPHERE)
    d_text_ob.remove()
    
    # ---> Handle
    h_text_ob, handle_texture = pr.create_texture(random.choice(HANDLE_TEXTURES))
    handle.remove_texture()
    handle.set_texture(handle_texture,TextureMappingMode.CYLINDER)
    h_text_ob.remove()

    target.set_position(position = list(np.random.uniform(position_min, position_max)),relative_to=handle)

    try :
        initial_image = vision_sensor.capture_rgb()
        initial_depth_left = depth_sensor_left.capture_depth()
        initial_depth_right = depth_sensor_right.capture_depth()
        # print(initial_image.shape)
        cv2.imwrite(path.join(DATASET_PATH,"initial_rgb_"+str(i)+".png"),initial_image*255)
        # cv2.imwrite(path.join(DATASET_PATH,"initial_depth_l_"+str(i)+".png"),initial_depth_left*255)
        # cv2.imwrite(path.join(DATASET_PATH,"initial_depth_r_"+str(i)+".png"),initial_depth_right*255)

        # print("-->",agent.get_joint_positions())
        move_arm_to_end(pr,agent,proximity_sensor,target.get_position()
                        ,target.get_quaternion(),target.get_orientation(),
                        True)
        # print(handle_fixed_position)
        # print(handle.get_position())
        handle_pos_eps = np.abs(handle_fixed_position - handle.get_position())
        handle_angle_eps = np.abs(handle_fixed_orientation - handle.get_orientation())

        robot_end_joint_positions = agent.get_joint_positions()

        if np.any(handle_pos_eps > 0.005 ) or np.any(handle_angle_eps > 0.3 ): 
            final_image = vision_sensor.capture_rgb()
            final_depth_left = depth_sensor_left.capture_depth()
            final_depth_right = depth_sensor_right.capture_depth()

            result_list.append([i,"failure",handle.get_orientation(),handle.get_position(),handle_pos_eps,handle_angle_eps,random_start_joint_positions,robot_end_joint_positions])

            cv2.imwrite(path.join(DATASET_PATH,"end_failure_rgb_"+str(i)+".png"),final_image*255)
            # cv2.imwrite(path.join(DATASET_PATH,"end_failure_depth_l_"+str(i)+".png"),final_depth_left*255)
            # cv2.imwrite(path.join(DATASET_PATH,"end_failure_depth_r_"+str(i)+".png"),final_depth_right*255)
            # print(str(i),",","failure",",",random_start_joint_positions,",",robot_end_joint_positions)
            
        else:
            result_list.append([i,"success",handle.get_orientation(),handle.get_position(),handle_pos_eps,handle_angle_eps,random_start_joint_positions,robot_end_joint_positions])
            # print(str(i),",","success",",",random_start_joint_positions,",",robot_end_joint_positions)
            final_image = vision_sensor.capture_rgb()
            final_depth_left = depth_sensor_left.capture_depth()
            final_depth_right = depth_sensor_right.capture_depth()
            
            cv2.imwrite(path.join(DATASET_PATH,"end_success_rgb_"+str(i)+".png"),final_image*255)
            # cv2.imwrite(path.join(DATASET_PATH,"end_success_depth_l_"+str(i)+".png"),final_depth_left*255)
            # cv2.imwrite(path.join(DATASET_PATH,"end_success_depth_r_"+str(i)+".png"),final_depth_right*255)


    except ConfigurationPathError as e:
        print("SKIPPING")
        continue

print("--->",len(result_list))
pickle.dump(result_list,text_file)

print("--------------------------------")

text_file.close()

pr.stop()
pr.shutdown()


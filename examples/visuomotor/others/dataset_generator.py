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
import uuid

#------------------------------
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_kinova3_door_env_random.ttt')

# ------------INIT-------------
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
# pr.set_simulation_timestep(0.2)
pr.start()





pr.stop()
pr.shutdown()
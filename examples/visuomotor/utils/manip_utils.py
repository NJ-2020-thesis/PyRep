import math
from pyrep import PyRep
from pyrep.errors import ConfigurationPathError


def move_arm(pr,agent,proximity,position, quaternion, orientation, ignore_collisions=False):
    arm_path = agent.get_path(position=position,
                            # euler=[0, math.radians(180), 0],
                            # orientation=orientation,
                            euler=[0, 0, 0],
                            ignore_collisions=ignore_collisions,
                            )
    arm_path.visualize()
    done = False
    while not done:
        done = arm_path.step()
        print(proximity.read())
        # print(agent.get_joint_positions())
        pr.step()

    arm_path.clear_visualization()

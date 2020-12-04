import math
from pyrep import PyRep
from pyrep.errors import ConfigurationPathError


def move_arm(pr,agent,proximity,position, quaternion, orientation, ignore_collisions=False):
    """
    Moves arm along with visualization of complete path.
    """
    arm_path = agent.get_path(position=position,
                            euler=[0, math.radians(180), 0],
                            # orientation=orientation,
                            # euler=[0, 0, 0],
                            ignore_collisions=ignore_collisions,
                            )
    print("Path found!!")
    # print(arm_path.values)
    arm_path.visualize()
    done = False
    i = 0
    while (i < 100) and (not done) :
        i += 1
        done = arm_path.step()
        print(proximity.read())
        # print(agent.get_joint_positions())
        pr.step()

    arm_path.clear_visualization()



def move_arm_to_end(pr,agent,proximity,position, quaternion, orientation, ignore_collisions=False):
    """
    Directly moves the arm to the end of the path.
    """
    DISTANCE = 0.05
    arm_path = agent.get_path(position=position,
                            euler=[0, math.radians(180), 0],
                            # orientation=orientation,
                            # euler=[0, 0, 0],
                            ignore_collisions=ignore_collisions,
                            )
    print("Path found!!")
    arm_path.set_to_end()
    if proximity.read() < DISTANCE:
        pass

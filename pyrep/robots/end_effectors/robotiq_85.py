from pyrep.robots.end_effectors.gripper import Gripper


class Robotiq85Gripper(Gripper):

    def __init__(self, count: int = 0):
        super().__init__(count, 'ROBOTIQ_85',
                         ['ROBOTIQ_85_Lactive1',
                          'ROBOTIQ_85_Ractive2'])

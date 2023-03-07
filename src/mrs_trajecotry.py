import numpy as np
from mrs_msgs.srv import PathSrv, PathSrvRequest, PathSrvResponse




def _send_fake_odometry(position: np.array):
    pass


def _send_fake_():
    pass





def fake_plan_trajectory(way_pts: np.array):
    """
    Main exported function from the file.
    The function is meant to be used only in testing environment. During a real (or simulated) flight,
    one can just normally call trajectory generation without all these fake functions
    :param way_pts: way points to visit 3
    :return:
    """
    pass
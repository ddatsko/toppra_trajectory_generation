import rospy
from mrs_msgs.srv import TransformReferenceSrv, TransformReferenceSrvRequest, TransformReferenceSrvResponse
from mrs_msgs.msg import Reference
from std_msgs.msg import Header
from typing import List
import math
import copy
import os
import numpy as np


def quaternion_to_roll_pitch_yaw(q) -> (float, float, float):
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    yaw = math.atan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
    pitch = math.asin(-2.0 * (qx * qz - qw * qy))
    roll = math.atan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    return roll, pitch, yaw


def _log_message(msg: str) -> str:
    return '[ToppraTrajectoryGeneration]: ' + msg


def log_err(msg: str):
    rospy.logerr(_log_message(msg))


def log_info(msg: str):
    rospy.loginfo(_log_message(msg))


def log_warn(msg: str):
    rospy.logwarn(_log_message(msg))


def get_parameter(param):
    param_name = rospy.get_name() + '/' + param
    if param_name not in rospy.get_param_names():
        log_err("Parameter " + param_name + " could not be leaded")
        raise KeyError
    return rospy.get_param(param_name)


def transform_reference(ref: Reference, header: Header, frame_id: str, seq: int = 0) -> Reference:
    uav_name = os.getenv('UAV_NAME')
    sp = rospy.ServiceProxy(f'/{uav_name}/control_manager/transform_reference', TransformReferenceSrv)

    req = TransformReferenceSrvRequest()
    req.frame_id = frame_id
    req.reference.header.seq = seq
    req.reference.header.frame_id = header.frame_id
    req.reference.reference = ref

    ref_t_resp: TransformReferenceSrvResponse = sp.call(req)

    if not ref_t_resp.success:
        log_err("Could not transform reference to gps_origin")

    # Set heading and altitude to the same ones as in the original reference
    ref_t = ref_t_resp.reference.reference
    ref_t.heading = ref.heading
    ref_t.position.z = ref.position.z

    return ref_t


def transform_to_gps_origin(path: List[Reference], header: Header) -> List[Reference]:
    seq = 0
    res = []
    for ref in path:
        ref_t = transform_reference(ref, header, 'gps_origin', seq)
        seq += 1
        res.append(ref_t)
    return res


def get_pose_in_frame(frame_id: str, pose_gps_origin: List[float]) -> List[float]:
    if frame_id == 'gps_origin':
        return copy.deepcopy(pose_gps_origin)
    else:
        origin_ref = Reference()
        origin_ref.position.x = pose_gps_origin[0]
        origin_ref.position.y = pose_gps_origin[1]
        origin_ref.position.z = pose_gps_origin[2]
        origin_ref.heading = pose_gps_origin[3]
        ref_t = transform_reference(origin_ref, Header(frame_id='gps_origin'), frame_id, 0)
        return [ref_t.position.x, ref_t.position.y, ref_t.position.z, ref_t.heading]


def get_angle(a, b, c):
    ang = math.radians(math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])))
    return ang + math.pi * 2 if ang < 0 else ang


def isclose(v1, v2, eps=1e-1):
    return abs(v1 - v2) < eps


def read_points_from_file(filename):
    with open(filename, 'r') as f:
        return np.array(list(map(lambda line: list(map(lambda x: float(x.strip()), line.split(','))) + [5.0, 0.0],
                                 filter(lambda x: x, f.readlines()))))


def waypoint_between_in_distance(w1, w2, distance):
    if np.linalg.norm(w2 - w1) < distance:
        return w2
    else:
        return w1 + (w2 - w1) / np.linalg.norm(w2 - w1) * distance

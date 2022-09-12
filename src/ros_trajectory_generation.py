#!/usr/bin/python

import rospy
from mrs_msgs.srv import TrajectoryReferenceSrv, TrajectoryReferenceSrvResponse, TrajectoryReferenceSrvRequest
from mrs_msgs.srv import PathSrv, PathSrvRequest, PathSrvResponse
from mrs_msgs.srv import TransformReferenceSrv, TransformReferenceSrvRequest, TransformReferenceSrvResponse
from mrs_msgs.msg import Reference
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import trajectory_generation
import os
import numpy as np
from typing import List
import math
import copy

UAV_NAME = 'uav1'

current_pose = []


def quaternion_to_roll_pitch_yaw(q) -> (float, float, float):
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    yaw = math.atan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
    pitch = math.asin(-2.0 * (qx * qz - qw * qy))
    roll = math.atan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    return roll, pitch, yaw


def odometry_callback(odom_msg: Odometry):
    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(odom_msg.pose.pose.orientation)
    position = odom_msg.pose.pose.position

    global current_pose
    current_pose = [position.x, position.y, position.z, yaw]


def log_err(msg: str):
    rospy.logerr('[ToppraTrajectoryGeneration]: ' + msg)


def log_info(msg: str):
    rospy.loginfo('[ToppraTrajectoryGeneration]: ' + msg)


def get_parameter(param):
    param_name = rospy.get_name() + '/' + param
    if param_name not in rospy.get_param_names():
        log_err("Parameter " + param_name + " could not be leaded")
        raise KeyError
    return rospy.get_param(param_name)


def transform_reference(ref: Reference, header: Header, frame_id: str, seq: int = 0) -> Reference:
    sp = rospy.ServiceProxy(f'/{UAV_NAME}/control_manager/transform_reference', TransformReferenceSrv)

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


def get_current_pose_in_frame(frame_id: str) -> List[float]:
    if frame_id == 'gps_origin':
        return copy.deepcopy(current_pose)
    else:
        origin_ref = Reference()
        origin_ref.position.x = current_pose[0]
        origin_ref.position.y = current_pose[1]
        origin_ref.position.z = current_pose[2]
        ref_t = transform_reference(origin_ref, Header(frame_id='gps_origin'), frame_id, 0)
        return [ref_t.position.x, ref_t.position.y, ref_t.position.z, ref_t.heading]


def service_generate_trajectory(req: PathSrvRequest) -> PathSrvResponse:
    try:
        sampling_dt = get_parameter('sampling_dt')
        # TODO: use not default constraints, but some from path request
        manager = trajectory_generation.TrajectoryGenerationManager(dof=4)

        if req.path.override_constraints:
            manager.max_speed = req.path.override_max_velocity_horizontal
            manager.max_acc = req.path.override_max_acceleration_horizontal
            manager.max_vert_speed = req.path.override_max_velocity_vertical
            manager.max_vert_acc = req.path.override_max_acceleration_vertical
            manager.max_speed_eps = get_parameter('max_speed_eps')
            manager.max_acc_eps = get_parameter('max_acc_eps')
            manager.max_heading_speed = get_parameter('heading_speed')
            manager.max_heading_acc = get_parameter('heading_acc')
            manager.distance_for_equal_limits = get_parameter('equal_limits_distance')
            manager.distance_between_added = get_parameter('distance_between_added_way_pts')
            manager.distance_around_turns = get_parameter('distance_between_added_and_turns')

        # Transform to gps_origin only of the original frame_id is bad
        if req.path.header.frame_id == 'latlon_origin':
            waypoints_gps = transform_to_gps_origin(req.path.points, req.path.header)
            res_frame_id = 'gps_origin'
        else:
            waypoints_gps = req.path.points
            res_frame_id = req.path.header.frame_id

        origin = get_current_pose_in_frame(res_frame_id)

        waypoints = np.array([origin] +
                             list(map(lambda p: [p.position.x, p.position.y, p.position.z, p.heading], waypoints_gps)))

        trajectory = manager.plan_trajectory(waypoints)

        log_info("Trajectory duration: " + str(trajectory.duration))

        ts_sample = np.arange(0, trajectory.duration, sampling_dt)
        qs_sample = trajectory(ts_sample)

        request = TrajectoryReferenceSrvRequest()
        request.trajectory.header.frame_id = res_frame_id
        request.trajectory.dt = sampling_dt
        request.trajectory.loop = False
        request.trajectory.fly_now = False
        request.trajectory.use_heading = True
        request.trajectory.points = []
        for point in qs_sample:
            reference = Reference()
            reference.heading = point[3]
            reference.position.z = point[2]
            reference.position.x = point[0]
            reference.position.y = point[1]
            request.trajectory.points.append(reference)

        run_type = os.getenv('RUN_TYPE')
        if run_type == 'simulation':
            # Using only one known UAV, so start trajectory following
            uav_name = os.getenv('UAV_NAME')
            trajectory_service = f'/{uav_name}/control_manager/trajectory_reference'

            sp = rospy.ServiceProxy(trajectory_service, TrajectoryReferenceSrv)
            resp = sp.call(request)

            log_info("Trajectory setting call result: " + resp.message)

            response = PathSrvResponse()
            response.success = resp.success
            response.message = resp.message
            return response
    except Exception as e:
        log_err("Trajectory not generated: " + str(e))
        return PathSrvResponse(success=False, message=str(e))
    return PathSrvResponse(success=True, message='No trajectory published')


def main():
    global UAV_NAME
    UAV_NAME = os.getenv('UAV_NAME')
    log_info("UAV name is set to " + UAV_NAME)

    rospy.init_node('toppra_trajectory_generation')

    log_info("Initialized node")

    subscriber = rospy.Subscriber(f'/{UAV_NAME}/odometry/odom_gps', Odometry, odometry_callback)
    service = rospy.Service('/toppra_trajectory_generation', PathSrv, service_generate_trajectory)

    rospy.spin()


if __name__ == '__main__':
    main()

#!/usr/bin/python

import rospy
import rosservice
from mrs_msgs.srv import TrajectoryReferenceSrv, TrajectoryReferenceSrvRequest
from mrs_msgs.srv import PathSrv, PathSrvRequest, PathSrvResponse
from mrs_msgs.msg import Reference
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, TriggerRequest
import trajectory_generation
import os
import numpy as np
from utils import get_parameter, log_info, log_err, quaternion_to_roll_pitch_yaw, get_pose_in_frame, transform_to_gps_origin, log_warn
from mrs_msgs.msg import DynamicsConstraints
import math

current_pose = []
current_constraints: DynamicsConstraints = DynamicsConstraints()
current_constraints_set = False


def constraints_callback(constraints: DynamicsConstraints):
    global current_constraints, current_constraints_set
    current_constraints_set = True
    current_constraints = constraints


def odometry_callback(odom_msg: Odometry):
    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(odom_msg.pose.pose.orientation)
    position = odom_msg.pose.pose.position

    global current_pose
    current_pose = [position.x, position.y, position.z, yaw]


def stop_trajectory_tracking():
    try:
        sp = rospy.ServiceProxy(f'{rospy.get_namespace()}control_manager/stop_trajectory_tracking', Trigger)
        req = TriggerRequest()
        sp.call(req)
    except Exception as e:
        log_warn(f"Could not stop trajectory tracking. Continuing planning. Error: {str(e)}")


def sleep_s(s: float):
    seconds = int(s)
    ns = int(math.modf(s)[0] * 1e9)
    rospy.sleep(rospy.Duration(seconds, ns))


def service_generate_trajectory(req: PathSrvRequest) -> PathSrvResponse:
    stop_trajectory_tracking()
    sleep_s(get_parameter('sleep_after_stop'))
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

            if get_parameter('constrain_with_constraint_manager'):
                if not current_constraints_set:
                    log_err("Current constraints still could not be read from constraint manager")
                    # TODO: use own exception class here
                    raise Exception("Current constraint not set")
                manager.max_speed = min(manager.max_speed, current_constraints.horizontal_speed)
                manager.max_acc = min(manager.max_acc, current_constraints.horizontal_acceleration)
                manager.max_vert_speed = min(manager.max_vert_speed, current_constraints.vertical_ascending_speed)
                manager.max_vert_acc = min(manager.max_vert_acc, current_constraints.vertical_ascending_acceleration)
                manager.max_heading_speed = min(manager.max_heading_speed, current_constraints.heading_speed)
                manager.max_heading_acc = min(manager.max_heading_acc, current_constraints.heading_acceleration)

        # Transform to gps_origin only of the original frame_id is bad
        if req.path.header.frame_id == 'latlon_origin':
            waypoints_gps = transform_to_gps_origin(req.path.points, req.path.header)
            res_frame_id = 'gps_origin'
        else:
            waypoints_gps = req.path.points
            res_frame_id = req.path.header.frame_id
        origin = get_pose_in_frame(res_frame_id, current_pose)

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
        request.trajectory.fly_now = req.path.fly_now
        request.trajectory.use_heading = True
        request.trajectory.points = []
        for point in qs_sample:
            reference = Reference()
            reference.position.x = point[0]
            reference.position.y = point[1]
            reference.position.z = point[2]
            reference.heading = point[3]

            request.trajectory.points.append(reference)

        run_type = os.getenv('RUN_TYPE')
        if run_type == 'simulation':
            # Using only one known UAV, so start trajectory following
            trajectory_service = f'{rospy.get_namespace()}control_manager/trajectory_reference'

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
    rospy.init_node('toppra_trajectory_generation')

    log_info("Initialized node")

    subscriber = rospy.Subscriber(f'{rospy.get_namespace()}odometry/odom_gps', Odometry, odometry_callback)
    service = rospy.Service(f'{rospy.get_namespace()}toppra_trajectory_generation', PathSrv, service_generate_trajectory)
    constraint_subscriber = rospy.Subscriber(f'{rospy.get_namespace()}control_manager/current_constraints', DynamicsConstraints, constraints_callback)


    rospy.spin()


if __name__ == '__main__':
    main()

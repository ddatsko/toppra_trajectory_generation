import numpy as np
import toppra as ta
from typing import List, Tuple, Type, Optional

import toppra.interpolator
from toppra.constraint import JointVelocityConstraintVarying
from acceleration_constraint_varying import JointAccelerationConstraintVarying
from toppra.constraint import DiscretizationType
import toppra.algorithm as algo
import math
from utils import get_angle
from path_manipulations import rotate_path, split_waypoints_into_sweeping_chunks, \
    move_connections_to_segments, add_waypoints_with_distance


# This class is intended to replace the one in trajectory_generation when working well
class TrajectoryGenerationManager2:
    # Using some random default values, which is convenient in testing
    # NOTE: the user should set these values from outside before trajectory planning
    max_speed = 8.9
    distance_between_added = 30
    distance_around_turns = 5
    distance_for_equal_limits = 15
    max_acc = 2
    max_speed_eps = 0.4
    max_speed_violation_eps = 1
    max_acc_eps = 0.5
    max_vert_speed = 2.0
    max_vert_acc = 1.0
    max_heading_speed = 1.0
    max_heading_acc = 1.0

    # Temporary variable for debug and plotting constraints
    # TODO: remove this when no plots of constraints are needed or make it private and accessed through a function
    constr_temp = []

    # list of all the waypoints with corresponding time stamp in format
    # [[...] TODO
    # TODO: remove waypoints and always use path.waypoints instead of it
    _waypoints: np.array = np.array([[], []])

    # Set of waypoints indices without turns
    _no_turn_waypoints_idx = set()
    _path = None
    _waypoints_max_constraints_distances: List[Tuple[float, float]] = []

    def __init__(self, dof: int = 4, log_info=print, log_error=print):
        """
        NOTE: you can pass "lambda x: pass" as loggers to switch the output off completely
        :param dof: Path degrees of freedom
        :param log_info: function for logging info data. Print by default
        :param log_error: function for logging errors. Print by default
        """
        self._waypoints_max_constraints_distances = []
        self._no_turn_waypoints_idx = set()
        self._waypoints = np.array([[], []])
        self.constr_temp = []
        self.log_info = log_info
        self.log_error = log_error
        if dof < 2:
            raise ValueError("Degree of freedom should be at least 2")
        self.dof = dof

    def _get_distances_between_waypoints(self, waypoint_idx_0: int, waypoint_idx_1: int, x: float) -> (float, float):
        """
        Get the distances between x and 2 waypoints (assuming that x is between of them)
        :param waypoint_idx_0: first waypoints index
        :param waypoint_idx_1: second waypoint index
        :param x: a value from 0 to 1 corresponding to the waypoints after toppra interpolation
        :return: distances to the first and second point correspondingly
        """
        p1, p2 = self._waypoints[1][waypoint_idx_0], self._waypoints[1][waypoint_idx_1]
        dist = np.linalg.norm(p2 - p1)
        if x == self._waypoints[0][waypoint_idx_0]:
            return 0, dist
        ratio_first = abs(
            (x - self._waypoints[0][waypoint_idx_0]) / (
                    self._waypoints[0][waypoint_idx_1] - self._waypoints[0][waypoint_idx_0]))
        return dist * ratio_first, dist * (1 - ratio_first)

    def _get_unit_vector(self, waypoints_idx_0: int, waypoints_idx_1: int) -> np.array:
        """
        Get the unit vector in direction from first point to the second one
        :param waypoints_idx_0: index of the first waypoint
        :param waypoints_idx_1: index of the second wayopint
        :return: unit vector
        """
        p1, p2 = self._waypoints[1][waypoints_idx_0][:2], self._waypoints[1][waypoints_idx_1][:2]
        return (p2 - p1) / np.linalg.norm(p2 - p1)

    def _custom_horizontal_constraints(self, x):
        """
        :param x: a value from 0 to 1 corresponding to the waypoints after toppra interpolation
        :return: Speed and acceleration constraints for first 2 DOF for that point taking into account
        distance to closest waypoints and the distance to them for relaxed constraints
        """
        # If x is not inside the path, just return constraints per axis
        if x < 0 or x > 1:
            return np.array([[-self.max_speed, self.max_speed],
                             [-self.max_speed, self.max_speed]]), \
                np.array([[-self.max_acc, self.max_acc],
                          [-self.max_acc, self.max_acc]])

        waypoints_labels = list(self._waypoints[0])
        for i in range(len(waypoints_labels)):
            if x < waypoints_labels[i]:
                prev_waypoint_idx = i - 1
                next_waypoint_idx = i
                break
        else:
            prev_waypoint_idx = len(waypoints_labels) - 2
            next_waypoint_idx = prev_waypoint_idx + 1

        # Change neighboring waypoints to turing ones
        while prev_waypoint_idx in self._no_turn_waypoints_idx:
            prev_waypoint_idx -= 1
        while next_waypoint_idx in self._no_turn_waypoints_idx:
            next_waypoint_idx += 1

        distances_to_waypoints = self._get_distances_between_waypoints(prev_waypoint_idx, next_waypoint_idx, x)

        # If the point is close enough to turning waypoints, set constraints to maximum on each axis
        if distances_to_waypoints[0] < self._waypoints_max_constraints_distances[prev_waypoint_idx][1] or \
                distances_to_waypoints[1] < self._waypoints_max_constraints_distances[next_waypoint_idx][0]:
            return np.array([[-self.max_speed - self.max_speed_eps, self.max_speed + self.max_speed_eps],
                             [-self.max_speed - self.max_speed_eps, self.max_speed + self.max_speed_eps]]), \
                np.array([[-self.max_acc - self.max_acc_eps, self.max_acc + self.max_acc_eps],
                          [-self.max_acc - self.max_acc_eps, self.max_acc + self.max_acc_eps]])

        # Otherwise, calculate the direction vector of the path between twi waypoints and set constraints in the way
        # that the maximum speed and acceleration can be reached only along that path segment
        x_speed, y_speed = map(abs, self._get_unit_vector(prev_waypoint_idx, next_waypoint_idx))
        # print(x_speed, y_speed)

        return np.array(
            [[-x_speed * self.max_speed - self.max_speed_eps, x_speed * self.max_speed + self.max_speed_eps],
             [-y_speed * self.max_speed - self.max_speed_eps, y_speed * self.max_speed + self.max_speed_eps]]), \
            np.array(
                [[-x_speed * self.max_acc - self.max_acc_eps, x_speed * self.max_acc + self.max_acc_eps],
                 [-y_speed * self.max_acc - self.max_acc_eps, y_speed * self.max_acc + self.max_acc_eps]])

    def _custom_constraints(self, x: float) -> (np.array, np.array):
        """
        Get the custom constraints in point x
        :param x: a value from 0 to 1 corresponding to the waypoints after toppra interpolation
        :return: speed and acceleration constraints in the point x for each DOF
        """
        horizontal_speed, horizontal_acc = self._custom_horizontal_constraints(x)
        additional_dof_speed = []
        additional_dof_acc = []

        # Add vertical speed and acceleration constraints if the path DOF >= 3 (x, y, z)
        if self.dof >= 3:
            additional_dof_speed.append([-self.max_vert_speed, self.max_vert_speed])
            additional_dof_acc.append([-self.max_vert_acc, self.max_vert_acc])

        # Add heading speed and acceleration constraints if the path DOF >= 4 (x, y, z, heading)
        if self.dof >= 4:
            additional_dof_speed.append([-self.max_heading_speed, self.max_heading_speed])
            additional_dof_acc.append([-self.max_heading_acc, self.max_heading_acc])

        return np.concatenate((horizontal_speed, additional_dof_speed)), \
            np.concatenate((horizontal_acc, additional_dof_acc))

    def _custom_speed_constraints(self, x: float) -> np.array:
        """
        Return custom constraints for the point x
        :param x: a value from 0 to 1 corresponding to the waypoints after toppra interpolation
        :return: speed constraints in all the degrees of freedom
        """
        constr = self._custom_constraints(x)[0]
        self.constr_temp.append([x, constr[0][1], constr[1][1]])
        return self._custom_constraints(x)[0]

    def _custom_acc_constraints(self, x):
        return self._custom_constraints(x)[1]

    def _plan_one_trajectory(self, way_pts=None):
        """
        Plan one trajectory using toppra and current constraints
        :param way_pts If None, self._path will be used. Otherwise, way points to visit
        :return: planned trajectory
        """
        pc_vel = ta.constraint.JointVelocityConstraintVarying(lambda x: self._custom_speed_constraints(x))
        pc_acc = JointAccelerationConstraintVarying([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                                    discretization_scheme=DiscretizationType.Collocation,
                                                    alim_func=lambda x: self._custom_acc_constraints(x))

        instance = algo.TOPPRA([pc_vel, pc_acc], way_pts, parametrizer='ParametrizeConstAccel')
        jnt_traj = instance.compute_trajectory()

        return jnt_traj

    def _plan_chunk_trajectory(self, chunk_way_pts, in_speed=0, out_speed=0):
        vlim, alim = self._custom_constraints(-1)

        pc_vel = ta.constraint.JointVelocityConstraint(vlim)
        pc_acc = ta.constraint.JointAccelerationConstraint(alim)

        instance = algo.TOPPRA([pc_vel, pc_acc], chunk_way_pts, parametrizer='ParametrizeConstAccel')
        # jnt_traj = instance.compute_trajectory(sd_start=in_speed, sd_end=out_speed)
        jnt_traj = instance.compute_trajectory()
        return jnt_traj

    def _no_constraints_init_distance(self, angle):
        """
        Compute the distance around the turn, at which the speed and acceleration will have constraints not dependent
        on the path segment rotation
        :param angle: angle of the turn
        :return: estimated distance before and after the waypoint, at which allowing max speed and acceleration in each axis should not lead to the
        constraints violation
        """
        # TODO: Make a more clever algorithm here, to not estimate too little or too much

        # If the angle is close to 0 or pi, set the distance to 0
        if min(abs(angle), abs(angle - math.pi)) < 0.1:
            return 0, 0

        # For now, just set very optimistic constraints, that for sure will have to be changed
        # Acceleration time
        t_acc = self.max_speed / self.max_acc
        s_acc = 0.5 * self.max_acc * t_acc ** 2
        return s_acc * 2, s_acc * 2

    def _constraints_violated(self, trajectory: toppra.interpolator.PolynomialPath) -> List[
        Tuple[int, Tuple[float, float]]]:
        """
        Find the points of speed constraint violation
        NOTE: the function returns the empty list if the speed constraints are satisfied
        NOTE: some distances to the closest violation may be INF
        :param trajectory: Planned trajectory from toppra
        :return: List of speed violation in format (<wypoint index>,
                                                    (distance to closest speed violation before the waypoint,
                                                    distance to closest speed violation after the waypoint))
        """
        # TODO: move this to some external parameter
        sampling_dt = 0.05
        ts = np.arange(0, trajectory.duration, sampling_dt)

        # Sample the trajectory position and speed
        pos_sample = trajectory(ts)
        vel_sample = trajectory(ts, 1)

        # Create the dict containing the closest distance to a constraint violation point from each waypoint
        closest_violation = {i: [float('inf'), float('inf')] for i in range(len(self._waypoints[0]))}

        turning_waypoints = [i for i in range(len(self._waypoints[0])) if i not in self._no_turn_waypoints_idx]

        # Go through each trajectory sample and check if constraints are violated
        # TODO: check here if no waypoint pass can be missed here due to sampling_dt or trajectory generation error
        current_waypoint = 0
        next_turning_waypoint_idx = 1
        next_waypoint = turning_waypoints[1]

        for i in range(len(ts)):
            pos = pos_sample[i]
            horizontal_vel = np.linalg.norm(vel_sample[i][:2])

            if horizontal_vel > self.max_speed + self.max_speed_violation_eps * np.sqrt(2):
                # print(f"VIOLATION: next_turning_waypoint_idx = {next_turning_waypoint_idx}, speed: {horizontal_vel}")
                # Update te previous and next waypoints violation distance
                current_waypoint_distance = np.linalg.norm(self._waypoints[1][current_waypoint] - pos)
                next_waypoint_distance = np.linalg.norm(self._waypoints[1][next_waypoint] - pos)

                closest_violation[current_waypoint][1] = min(closest_violation[current_waypoint][1],
                                                             current_waypoint_distance)
                closest_violation[next_waypoint][0] = min(closest_violation[next_waypoint][0], next_waypoint_distance)

            # If the point is close to the next turning waypoint, change the current segment
            # TODO: check if no waypoints are missed here which will lead to a completely broken performance
            if np.linalg.norm(self._waypoints[1][next_waypoint] - pos) < 0.5 and next_turning_waypoint_idx + 1 < len(
                    turning_waypoints):
                current_waypoint = next_waypoint
                next_turning_waypoint_idx += 1
                next_waypoint = turning_waypoints[next_turning_waypoint_idx]

        # If all the turning points were not traversed -- some of them were missed because of sampling dt.
        # This completely breaks the algorithm
        if next_turning_waypoint_idx != len(turning_waypoints) - 1:
            self.log_error("ERROR: not all turning waypoints checked while checking the trajectoty for violations")
            # TODO: make an owm error type for raising here
            raise RuntimeError("Not all turing waypoints checked while checking the trajectory for violations")

        return [(idx, violation) for idx, violation in closest_violation.items() if
                violation != [float('inf'), float('inf')]]

    def _update_waypoint_constraints(self, violation: List[Tuple[int, Tuple[float, float]]]) -> None:
        """
        Update the distances with no constraints around each waypoint based on the distances of the closest speed
        constraint violation to each waypoint
        :param violation: List of speed violation in format (<wypoint index>,
                                                            (distance to the closest speed violation before the waypoint,
                                                            distance to the closest speed violation after the waypoint))
        """
        # TODO: check if this condition is not too strict
        for idx, violation_distance in violation:
            # Using the fact that there was no violation closer than "violation_distance", setting this threshold should be enough
            new_left_const = min(self._waypoints_max_constraints_distances[idx][0], violation_distance[0])
            new_right_const = min(self._waypoints_max_constraints_distances[idx][1], violation_distance[1])

            # Update new constraints as a mean value between old constraints distances and violation distance
            self._waypoints_max_constraints_distances[idx] = [
                (new_left_const + self._waypoints_max_constraints_distances[idx][0]) / 2,
                (new_right_const + self._waypoints_max_constraints_distances[idx][1]) / 2]

    def plan_trajectory(self, way_pts) -> Optional[ta.interpolator.AbstractGeometricPath]:
        """
        Main function for trajectory planning
        Assuming that the UAV is located in waypoints[0] initially with 0 speed
        :param way_pts: way points to visit
        :param logger: function for logging info messages
        :return: Planned trajectory from toppra
        """
        # TODO: check if linspae here is a good idea and if the numbers should not correspond to the distance between waypoints
        ss = np.linspace(0, 1, way_pts.shape[0])
        self._path = ta.SplineInterpolator(ss, way_pts)
        self._waypoints = self._path.waypoints

        # Generate initial constraints
        self._waypoints_max_constraints_distances.append((0, 0))
        for i in range(1, len(self._waypoints[0]) - 1):
            turning_angle = get_angle(*self._waypoints[1][i - 1: i + 2])
            # TODO: maybe, introduce some parameter for the eps here
            if abs(turning_angle - math.pi) < 0.1:
                self._no_turn_waypoints_idx.add(i)
            self._waypoints_max_constraints_distances.append(self._no_constraints_init_distance(turning_angle))

        self._waypoints_max_constraints_distances.append((0, 0))

        self.log_info("Planning the initial trajectory...")

        trajectory = self._plan_one_trajectory(self._path)
        replan_counter = 0

        # print("Initial constraint distances: ", self._waypoints_max_constraints_distances)

        while violation := self._constraints_violated(trajectory):
            # TODO: introduce some parameter for the maximum number of replannings here
            if replan_counter >= 5:
                break
            replan_counter += 1

            self.constr_temp = []
            self.log_info(f"Replanning the trajectory. Iteration: {replan_counter}")
            self._update_waypoint_constraints(violation)

            # print("Waypoints constraints distances after replanning: ", self._waypoints_max_constraints_distances)

            trajectory = self._plan_one_trajectory(self._path)
        else:
            self.log_info("Trajectory planned successfully with constraints not violated")
            return trajectory

        self.log_info("Failed to produce a trajectory not violating the constraints...")
        return trajectory

    # def _match_2_trajectories(self, trajectory1: (np.array, np.array, np.array), trajectory2: (np.array, np.array, np.array), sampling_dt=0.1) -> (int, int):
    #     """
    #     Find two points from different trajectories, at which the merging will be the best
    #     (TODO: describe the criteria of "better" merging point)
    #     :param sampling_dt: time difference between each two consecutive points in trajectories
    #     :return:
    #     """
    #     # At this distance points are considered to be "close", meaning almost in the same location
    #     close_eps = 2 * sampling_dt * np.linalg.norm(np.array([self.max_speed, self.max_speed, self.max_vert_speed, self.max_heading_speed]))
    #
    #     # Pointer for trajectory 1 and trajectory 2
    #     i1, i2 = trajectory1[0].shape[0] - 1, 0
    #     while np.linalg.norm(trajectory1[0][i1] - trajectory2[0][i2]) > close_eps and i1 >= 0:
    #         # print(np.linalg.norm(trajectory1[0][i1] - trajectory2[0][i2]))
    #         i1 -= 1
    #
    #     if i1 < 0:
    #         # TODO: replace with a custom error in future
    #         raise RuntimeError("Could not find a matching point in trajectories")
    #
    #     while i1 != trajectory1[0].shape[0] - 1:
    #         dist = np.linalg.norm(trajectory1[0][i1] - trajectory2[0][i2])
    #         # print(f"Point: {trajectory1[0][i1]}, {trajectory2[0][i2]}, distance: {dist}")
    #
    #         d1 = np.linalg.norm(trajectory1[0][i1] - trajectory1[0][-1])
    #         d2 = np.linalg.norm(trajectory2[0][i2] - trajectory1[0][-1])
    #         # print(f"d1: {d1}, d2: {d2}")
    #
    #         if d2 < d1:
    #             i1 += 1
    #             continue
    #
    #         if d2 > np.linalg.norm(trajectory2[0][i2 + 1] - trajectory1[0][-1]):
    #             i2 += 1
    #             continue
    #
    #         break
    #
    #
    #
    # def merge_trajectories(self, trajectories):
    #     ts = np.arange(0, trajectories[0].duration, 0.1)
    #
    #     # Generate initial trajectory parameters from the first trajectory in the list
    #     res_positions = trajectories[0](ts)
    #     res_velocities = trajectories[0](ts, 1)
    #     res_accelerations = trajectories[0](ts, 2)
    #
    #     for i in range(1, len(trajectories)):
    #         ts = np.arange(0, trajectories[i].duration, 0.1)
    #         positions = trajectories[i](ts)
    #         velocities = trajectories[i](ts)
    #         accelerations = trajectories[i](ts)
    #
    #         self._match_2_trajectories((res_positions, res_velocities, res_accelerations), (positions, velocities, accelerations))
    #

    def plan_trajectory_by_chunks(self, way_pts, sampling_dt):
        """
        Plan trajectory by splitting it into chunks of "sweeping" and gluing them together
        :param way_pts: List of waypoints. The decomposition into chunks is performed internally
        :param sampling_dt time difference between samples in the produces trajectory
        :return: trajectory as a sequence of waypoints with time difference between them equal to sampling_dt
        """
        way_pts = np.array(way_pts)
        # way_pts =
        # If the array is 2d meaning that way_pts consists way points only, decompose it into chunks internally
        split = split_waypoints_into_sweeping_chunks(way_pts)
        # moved = move_connections_to_segments(self.max_speed, self.max_acc, split)

        # print(moved[0].waypoints)
        # print("======")
        # print(moved[1].waypoints)

        full_trajectory = None
        for rotation, waypoints in split:

            waypoints = add_waypoints_with_distance(waypoints, 5)
            path = rotate_path(waypoints, -rotation)

            import matplotlib.pyplot as plt
            interpolated = ta.SplineInterpolator(np.linspace(0, 1, path.shape[0]), path)
            ss_sam = np.linspace(0, 1, path.shape[0] * 10)
            vel = interpolated(ss_sam, order=1)
            plt.plot(vel[:, 0])
            plt.plot(vel[:, 1])
            plt.show()

            trajectory = self._plan_chunk_trajectory(
                ta.SplineInterpolator(np.linspace(0, 1, path.shape[0]), path),
                in_speed=0,
                out_speed=0)
            if trajectory is None:
                raise Exception("No trajectory produced. TODO: write the code to not whrow any exception here")

            ts = np.arange(0, trajectory.duration, sampling_dt)
            position_samples = trajectory(ts)
            velocity_samples = trajectory(ts, 1)
            acceleration_samples = trajectory(ts, 2)

            plt.plot(position_samples[:, 0], position_samples[:, 1])
            plt.plot(path[:, 0], path[:, 1])
            plt.show()

            # position_samples = rotate_path(position_samples, chunk.rotation)
            # velocity_samples = rotate_path(velocity_samples, chunk.rotation)
            # acceleration_samples = rotate_path(acceleration_samples, chunk.rotation)

            position_samples = rotate_path(position_samples, rotation)

            chunk_trajectory = (position_samples, velocity_samples, acceleration_samples)

            if full_trajectory is None:
                full_trajectory = chunk_trajectory
            else:
                full_trajectory = tuple(
                    map(lambda t: np.concatenate((t[0], t[1])), zip(full_trajectory, chunk_trajectory)))

        return full_trajectory


if __name__ == '__main__':
    # Test functions here
    way_pts = np.array([[-2, 0], [-1, 0], [0, 0], [1, 1], [2, 2]])
    split = split_waypoints_into_sweeping_chunks(way_pts)
    print(split)
    for angle, path in split:
        print("---------")
        new_p = rotate_path(path, angle)
        print(new_p)

# [[1.18072096e+03 1.13686838e-13 0.00000000e+00 0.00000000e+00]
#  [8.57483500e+02 1.17375050e+03 0.00000000e+00 0.00000000e+00]
#  [1.00195000e+03 1.40638000e+03 0.00000000e+00 0.00000000e+00]] 1.015067480620981

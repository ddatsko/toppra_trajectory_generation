import numpy as np
import toppra as ta
from typing import List, Tuple

import toppra.interpolator
from toppra.constraint import JointVelocityConstraintVarying
from acceleration_constraint_varying import JointAccelerationConstraintVarying
from toppra.constraint import DiscretizationType
import toppra.algorithm as algo
import math
from utils import get_angle


# This class is intended to replace the one in trajectory_generation when working well
class TrajectoryGenerationManager2:
    # Using some random default values, which is convenient in testing
    max_speed = 8.9
    distance_between_added = 30
    distance_around_turns = 5
    distance_for_equal_limits = 15
    max_acc = 2
    max_speed_eps = 0.4
    max_acc_eps = 0.5
    max_vert_speed = 2.0
    max_vert_acc = 1.0
    max_heading_speed = 1.0
    max_heading_acc = 1.0

    constr_temp = []

    class WaypointConstraint:
        def __init__(self, x_speed, x_acc=0, y_speed=0, y_acc=0, z_speed=0, z_acc=0, heading_speed=0, heading_acc=0):
            self.x_speed = x_speed
            self.x_acc = x_acc
            self.y_speed = y_speed
            self.y_acc = y_acc
            self.z_speed = z_speed
            self.z_acc = z_acc
            self.heading_speed = heading_speed
            self.heading_acc = heading_acc

        def get_constraints_list(self, dof) -> (list, list):
            constraints = ([self.x_speed, self.y_speed, self.z_speed, self.heading_speed],
                           [self.x_acc, self.y_acc, self.z_acc, self.heading_acc])
            return constraints[0][:dof], constraints[1][:dof]

    # list of all the waypoints with corresponding time stamp in format
    # [[...] TODO
    # TODO: remove waypoints and always use path.waypoints instead of it
    _waypoints: np.array = np.array([[], []])
    _path = None
    _current_waypoints_constraints: List[WaypointConstraint] = []
    _waypoints_max_constraints_distances: List[Tuple[float, float]] = []

    def __init__(self, dof: int = 4):
        if dof < 2:
            raise ValueError("Degree of freedom should be at least 2")
        self.dof = dof

    def _get_waypoint_constraints(self, waypoint_idx: int) -> (np.array, np.array):
        if waypoint_idx < 0 or waypoint_idx >= len(self._current_waypoints_constraints):
            raise IndexError(f"Waypoint with index {waypoint_idx} does not exist")
        constraint = self._current_waypoints_constraints[waypoint_idx]

    def _get_distances_between_waypoints(self, waypoint_idx_0, waypoint_idx_1, x):
        p1, p2 = self._waypoints[1][waypoint_idx_0], self._waypoints[1][waypoint_idx_1]
        dist = np.linalg.norm(p2 - p1)
        if x == self._waypoints[0][waypoint_idx_0]:
            return 0, dist
        ratio_first = abs(
            (x - self._waypoints[0][waypoint_idx_0]) / (
                    self._waypoints[0][waypoint_idx_1] - self._waypoints[0][waypoint_idx_0]))
        return dist * ratio_first, dist * (1 - ratio_first)

    def _get_unit_vector(self, waypoints_idx_0, waypoints_idx_1):
        p1, p2 = self._waypoints[1][waypoints_idx_0][:2], self._waypoints[1][waypoints_idx_1][:2]
        return (p2 - p1) / np.linalg.norm(p2 - p1)

    def _custom_horizontal_constraints(self, x):
        """
        :param x: value from 0 to 1 representing a path point
        :return: Speed and acceleration constraints for that point taking into account
        distance to closest waypoints and the distance to them for relaxed constraints
        """
        waypoints_labels = list(self._waypoints[0])
        for i in range(len(waypoints_labels)):
            if x < waypoints_labels[i]:
                prev_waypoint_idx = i - 1
                break
        else:
            prev_waypoint_idx = len(waypoints_labels) - 2

        distances_to_waypoints = self._get_distances_between_waypoints(prev_waypoint_idx, prev_waypoint_idx + 1, x)

        # If the point is close enough to some of waypoints, set constraints to maximum on each axis
        if distances_to_waypoints[0] < self._waypoints_max_constraints_distances[prev_waypoint_idx][1] or \
                distances_to_waypoints[1] < self._waypoints_max_constraints_distances[prev_waypoint_idx + 1][0]:
            return np.array([[-self.max_speed, self.max_speed], [-self.max_speed, self.max_speed]]), \
                   np.array([[-self.max_acc, self.max_acc], [-self.max_acc, self.max_acc]])

        # Otherwise, calculate the direction vector of the path between twi waypoints and set constraints in the way
        # that the maximum speed and acceleration can be reached only along that path segment
        x_speed, y_speed = map(abs, self._get_unit_vector(prev_waypoint_idx, prev_waypoint_idx + 1))

        return np.array(
            [[-x_speed * self.max_speed - self.max_speed_eps, x_speed * self.max_speed + self.max_speed_eps],
             [-y_speed * self.max_speed - self.max_speed_eps, y_speed * self.max_speed + self.max_speed_eps]]), \
               np.array(
                   [[-x_speed * self.max_acc - self.max_acc_eps, x_speed * self.max_acc + self.max_acc_eps],
                    [-y_speed * self.max_acc - self.max_acc_eps, y_speed * self.max_acc + self.max_acc_eps]])

    def _custom_constraints(self, x):
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

        # print(x, np.concatenate((horizontal_speed, additional_dof_speed)), \
        #        np.concatenate((horizontal_acc, additional_dof_acc)))

        return np.concatenate((horizontal_speed, additional_dof_speed)), \
               np.concatenate((horizontal_acc, additional_dof_acc))

    def _custom_speed_constraints(self, x):
        constr = self._custom_constraints(x)[1]
        self.constr_temp.append([x, constr[0][1], constr[1][1]])
        return self._custom_constraints(x)[0]

    def _custom_acc_constraints(self, x):
        return self._custom_constraints(x)[1]

    def _plan_one_trajectory(self):
        pc_vel = ta.constraint.JointVelocityConstraintVarying(lambda x: self._custom_speed_constraints(x))
        pc_acc = JointAccelerationConstraintVarying([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                                    discretization_scheme=DiscretizationType.Collocation,
                                                    alim_func=lambda x: self._custom_acc_constraints(x))

        instance = algo.TOPPRA([pc_vel, pc_acc], self._path, parametrizer='ParametrizeConstAccel')
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

        # If the angle is close to 0 or 2*pi, set the distance to 0
        if min(abs(angle), abs(angle - math.pi)) < 0.1:
            return 0, 0

        # For now, just set very optimistic constraints, that for sure will have to be changed
        # Acceleration time
        t_acc = self.max_speed / self.max_acc
        s_acc = 0.5 * self.max_acc * t_acc ** 2

        print(angle, s_acc)
        return s_acc * 2, s_acc * 2

    def _constraints_violated(self, trajectory: toppra.interpolator.PolynomialPath) -> List[Tuple[int, float]]:
        # TODO
        sampling_dt = 0.4
        ts = np.arange(0, trajectory.duration, sampling_dt)

        # Sample the trajectory position and speed
        pos_sample = trajectory(ts)
        vel_sample = trajectory(ts, 1)

        # Create the dict containing the closest distance to a constraint violation point from each waypoint
        closest_violation = {i: float('inf') for i in range(len(self._waypoints[0]))}

        # Go through each trajectory sample and check if constraints are violated
        # TODO: check here if no waypoint pass can be missed here due to sampling_dt or trajectory generation error
        current_waypoint = 0
        next_waypoint = 1

        for i in range(len(ts)):
            pos = pos_sample[i]
            horizontal_vel = np.norm(vel_sample[i][:2])

            if horizontal_vel > self.max_speed + self.max_speed_eps:
                # Update te previous waypoint violation distance
                current_waypoint_distance = np.norm(self._waypoints[current_waypoint] - pos)
                next_waypoint_distance = np.norm(self._waypoints[next_waypoint] - pos)

                closest_violation[current_waypoint] = min(closest_violation[current_waypoint],
                                                          current_waypoint_distance)
                closest_violation[next_waypoint] = min(closest_violation[next_waypoint], next_waypoint_distance)

        return [(idx, violation) for idx, violation in closest_violation if closest_violation != float('inf')]

    def _update_waypoint_constraints(self, trajectory):
        # TODO
        pass

    def plan_trajectory(self, way_pts):
        # TODO: check if linspae here is a good idea and if the numbers should not correspond to the distance between waypoints
        ss = np.linspace(0, 1, way_pts.shape[0])
        self._path = ta.SplineInterpolator(ss, way_pts)
        self._waypoints = self._path.waypoints

        # Generate initial constraints
        self._waypoints_max_constraints_distances.append((0, 0))
        for i in range(1, len(self._waypoints[0]) - 1):
            self._waypoints_max_constraints_distances.append(self._no_constraints_init_distance(get_angle(
                self._waypoints[1][i - 1],
                self._waypoints[1][i],
                self._waypoints[1][i + 1]
            )))
        self._waypoints_max_constraints_distances.append((0, 0))

        trajectory = self._plan_one_trajectory()
        while violation := self._constraints_violated(trajectory):
            self._update_waypoint_constraints(trajectory)
            trajectory = self._plan_one_trajectory()

        return trajectory

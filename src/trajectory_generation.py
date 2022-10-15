import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import math
from acceleration_constraint_varying import JointAccelerationConstraintVarying
from toppra.constraint import DiscretizationType
from enum import Enum
from utils import isclose, get_angle, waypoint_between_in_distance

SQRT_2 = 2 ** 0.5


class TrajectoryGenerationManager:
    # Using some random default values, which is convenient in testing
    max_speed = 8.9
    distance_between_added = 30
    distance_around_turns = 5
    distance_for_equal_limits = 15
    max_acc = 2
    max_speed_eps = 0.2
    max_acc_eps = 0.2
    max_vert_speed = 2.0
    max_vert_acc = 1.0
    max_heading_speed = 1.0
    max_heading_acc = 1.0
    waypoints: np.array = np.array([[], []])

    def __init__(self, dof=2):
        """
        :param dof: Degrees of freedom of the path. Must be at least 2
        """
        self.init_waypoints_idx = set()
        if dof < 2:
            raise ValueError("DOF should be at least 2")
        self.dof = dof

    def add_waypoints(self, init_waypoints):
        res = []
        self.init_waypoints_idx = set()
        for i in range(len(init_waypoints) - 1):
            if i == 0 or not isclose(get_angle(init_waypoints[i - 1], init_waypoints[i], init_waypoints[i + 1]),
                                     math.pi):
                self.init_waypoints_idx.add(len(res))
            res.append(init_waypoints[i])
            distance = np.linalg.norm(init_waypoints[i + 1] - init_waypoints[i])
            if distance < 2 * self.distance_around_turns:
                continue
            number_of_added = math.ceil((distance - 2 * self.distance_around_turns) / self.distance_between_added)
            first_added_shift = self.distance_around_turns + (
                    distance - 2 * self.distance_around_turns - (number_of_added - 1) * self.distance_between_added) / 2
            for j in range(number_of_added):
                res.append(waypoint_between_in_distance(init_waypoints[i], init_waypoints[i + 1], first_added_shift))
                first_added_shift += self.distance_between_added
        res.append(init_waypoints[-1])
        return np.array(res)

    def get_extra_dimensions_constraints(self):
        res = [[], []]
        # Add z axis constraints
        if self.dof > 2:
            res[0].append([-self.max_vert_speed, self.max_vert_speed])
            res[1].append([-self.max_vert_acc, self.max_vert_acc])

        # Add heading constraints
        if self.dof > 3:
            res[0].append([-self.max_heading_speed, self.max_heading_speed])
            res[1].append([-self.max_heading_acc, self.max_heading_acc])

        return np.array(res)

    def custom_acceleration(self, x):
        return self.custom_constraints(x)[1]

    def custom_speed(self, x):
        return self.custom_constraints(x)[0]

    def custom_constraints(self, x):
        # Get the closest point to the given path point from path waypoints
        values = list(self.waypoints[0])
        closest_idx = 0
        for i in range(1, len(values)):
            if values[i] >= x:
                closest_idx = i
                break
        else:
            closest_idx = len(values) - 1

        # TODO: Check if this works fine
        # If the closest index is a turning point, make the speed constraints be equal in each axis
        distance_between_neighbors = self.get_distances_between_waypoints(closest_idx - 1, closest_idx, x)

        if (closest_idx in self.init_waypoints_idx and distance_between_neighbors[
            1] < self.distance_for_equal_limits) or \
                ((closest_idx - 1) in self.init_waypoints_idx and distance_between_neighbors[
                    0] < self.distance_for_equal_limits):
            res = [np.array([[-self.max_speed, self.max_speed], [-self.max_speed, self.max_speed]]) / SQRT_2,
                   np.array([[-self.max_acc, self.max_acc], [-self.max_acc, self.max_acc]]) / SQRT_2]
        else:
            x_speed, y_speed = self.get_unit_vector(closest_idx - 1, closest_idx)
            x_speed = abs(x_speed)
            y_speed = abs(y_speed)

            res = [
                np.array(
                    [[-x_speed * self.max_speed - self.max_speed_eps, x_speed * self.max_speed + self.max_speed_eps],
                     [-y_speed * self.max_speed - self.max_speed_eps, y_speed * self.max_speed + self.max_speed_eps]]),
                np.array([[-x_speed * self.max_acc - self.max_acc_eps, x_speed * self.max_acc + self.max_acc_eps],
                          [-y_speed * self.max_acc - self.max_acc_eps, y_speed * self.max_acc + self.max_acc_eps]])]

        # print(res)
        extra_constraints = self.get_extra_dimensions_constraints()
        # print(extra_constraints)
        if extra_constraints.size != 0:
            res[0] = np.concatenate((res[0], extra_constraints[0]))
            res[1] = np.concatenate((res[1], extra_constraints[1]))

        return res

    def plan_trajectory(self, way_pts):
        # way_pts = self.add_waypoints(way_pts)

        ss = np.linspace(0, 1, way_pts.shape[0])

        path = ta.SplineInterpolator(ss, way_pts)

        self.waypoints = path.waypoints

        pc_vel = constraint.JointVelocityConstraintVarying(lambda x: self.custom_speed(x))
        pc_acc = JointAccelerationConstraintVarying([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                                    discretization_scheme=DiscretizationType.Collocation,
                                                    alim_func=lambda x: self.custom_acceleration(x))

        instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer='ParametrizeConstAccel')
        jnt_traj = instance.compute_trajectory()

        return jnt_traj

    def get_distances_between_waypoints(self, waypoint_idx_0, waypoint_idx_1, x):
        p1, p2 = self.waypoints[1][waypoint_idx_0], self.waypoints[1][waypoint_idx_1]
        dist = np.linalg.norm(p2 - p1)
        if x == self.waypoints[0][waypoint_idx_0]:
            return 0, dist
        ratio_first = abs(
            (x - self.waypoints[0][waypoint_idx_0]) / (
                        self.waypoints[0][waypoint_idx_1] - self.waypoints[0][waypoint_idx_0]))
        return dist * ratio_first, dist * (1 - ratio_first)

    def get_unit_vector(self, waypoints_idx_0, waypoints_idx_1):
        p1, p2 = self.waypoints[1][waypoints_idx_0][:2], self.waypoints[1][waypoints_idx_1][:2]
        return (p2 - p1) / np.linalg.norm(p2 - p1)

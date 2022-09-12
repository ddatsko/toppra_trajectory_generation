import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import math
from acceleration_constraint_varying import JointAccelerationConstraintVarying
from toppra.constraint import DiscretizationType

waypoints: np.array = np.array([[], []])
DISTANCE_BETWEEN_ADDED = 30
DISTANCE_AROUND_TURNS = 5
DISTANCE_FOR_EQUAL_LIMITS = 15
MAX_SPEED = 8.9  # Should be float
MAX_SPEED_EPS = 0.2
MAX_ACC = 2
MAX_ACC_EPS = 0.2
SQRT_2 = 2 ** 0.5
SAMPLING_DT = 0.1

MAX_VERT_SPEED = 2.0
MAX_VERT_ACC = 1.0

MAX_HEADING_SPEED = 1.0
MAX_HEADING_ACC = 1.0

init_waypoints_idx = set()

turns = [0, 0]


def get_angle(a, b, c):
    ang = math.radians(math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])))

    res = ang + math.pi * 2 if ang < 0 else ang

    # print(a, b, c, math.degrees(res))
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


def get_distances_between_waypoints(waypoint_idx_0, waypoint_idx_1, x):
    global waypoints
    p1, p2 = waypoints[1][waypoint_idx_0], waypoints[1][waypoint_idx_1]
    dist = np.linalg.norm(p2 - p1)
    if x == waypoints[0][waypoint_idx_0]:
        return 0, dist
    ratio_first = abs(
        (x - waypoints[0][waypoint_idx_0]) / (waypoints[0][waypoint_idx_1] - waypoints[0][waypoint_idx_0]))
    # print("Ratios: ", waypoints[0][waypoint_idx_0], waypoints[0][waypoint_idx_1], x, ratio_first)
    return dist * ratio_first, dist * (1 - ratio_first)


def get_unit_vector(waypoints_idx_0, waypoints_idx_1):
    global waypoints
    p1, p2 = waypoints[1][waypoints_idx_0][:2], waypoints[1][waypoints_idx_1][:2]
    return (p2 - p1) / np.linalg.norm(p2 - p1)


class TrajectoryGenerationManager:
    def __init__(self, dof=2):
        """
        :param dof: Degrees of freedom of the path. Must be at least 2
        """
        self.distance_between_added = DISTANCE_BETWEEN_ADDED
        self.distance_around_turns = DISTANCE_AROUND_TURNS
        self.distance_for_equal_limits = DISTANCE_FOR_EQUAL_LIMITS
        self.max_speed = MAX_SPEED
        self.max_acc = MAX_ACC
        self.max_speed_eps = MAX_SPEED_EPS
        self.max_acc_eps = MAX_ACC_EPS
        self.max_vert_speed = MAX_VERT_SPEED
        self.max_vert_acc = MAX_VERT_ACC
        self.max_heading_speed = MAX_HEADING_SPEED
        self.max_heading_acc = MAX_HEADING_ACC

        if dof < 2:
            raise ValueError("DOF should be at least 2")
        self.dof = dof

    def add_waypoints(self, init_waypoints):
        res = []
        global init_waypoints_idx
        init_waypoints_idx = set()
        for i in range(len(init_waypoints) - 1):
            if i == 0 or not isclose(get_angle(init_waypoints[i - 1], init_waypoints[i], init_waypoints[i + 1]),
                                     math.pi):
                init_waypoints_idx.add(len(res))
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

    def custom_constraints(self, x, constraint_type: str = 'velocity'):
        global waypoints
        # Get the closest point to the given path point from path waypoints
        values = list(waypoints[0])
        closest_idx = 0
        for i in range(1, len(values)):
            if values[i] >= x:
                closest_idx = i
                break
        else:
            closest_idx = len(values) - 1

        # TODO: Check if this works fine
        # If the closest index is a turning point, make the speed constraints be equal in each axis
        distance_between_neighbors = get_distances_between_waypoints(closest_idx - 1, closest_idx, x)

        if (closest_idx in init_waypoints_idx and distance_between_neighbors[1] < self.distance_for_equal_limits) or \
                ((closest_idx - 1) in init_waypoints_idx and distance_between_neighbors[
                    0] < self.distance_for_equal_limits):
            turns[0] += 1
            res = [np.array([[-self.max_speed, self.max_speed], [-self.max_speed, self.max_speed]]) / SQRT_2,
                   np.array([[-self.max_acc, self.max_acc], [-self.max_acc, self.max_acc]]) / SQRT_2]
        else:
            turns[1] += 1
            x_speed, y_speed = get_unit_vector(closest_idx - 1, closest_idx)
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

        if constraint_type == 'velocity':
            return res[0]
        elif constraint_type == 'acceleration':
            return res[1]
        else:
            raise ValueError(f"Wrong constraint type {constraint_type}")

    def plan_trajectory(self, way_pts):
        # constraint_manager = TrajectoryGenerationManager()

        way_pts = self.add_waypoints(way_pts)

        ss = np.linspace(0, 1, way_pts.shape[0])

        path = ta.SplineInterpolator(ss, way_pts)

        global waypoints
        waypoints = path.waypoints

        pc_vel = constraint.JointVelocityConstraintVarying(lambda x: self.custom_constraints(x, 'velocity'))
        pc_acc = JointAccelerationConstraintVarying([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                                    discretization_scheme=DiscretizationType.Collocation,
                                                    alim_func=lambda x: self.custom_constraints(x,
                                                                                                'acceleration'))

        instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer='ParametrizeConstAccel')
        jnt_traj = instance.compute_trajectory()

        return jnt_traj


import numpy as np
from typing import List, Tuple
import math
from math import cos, sin, ceil
from utils import waypoint_between_in_distance

ROTATION_VIOLATION_EPS = math.pi / 36  # 5 degrees


class PathChunk:
    """
    Just a simple class for storing some data and not using Tuple[...]
    """

    def __init__(self, in_speed, out_speed, rotation, waypoints):
        self.in_speed = in_speed
        self.out_speed = out_speed
        self.rotation = rotation
        self.waypoints = waypoints


class PathManipulationError(Exception):
    pass


class PathSplitError(PathManipulationError):
    pass


def _segment_rotation(p1, p2):
    # As there may be more degrees of freedom in the path, take only first 2 -- 2 horizontal axes
    x, y = p1[0:2] - p2[0:2]
    return (math.atan2(y, x) + math.pi) % math.pi


def _get_max_final_speed(distance, acc):
    return (2 * acc * distance) ** 0.5


def _cut_segment(s0, s1, dist_from_center=10) -> (np.array, np.array, np.array):
    middle = (s0 + s1) / 2
    if np.linalg.norm(s0 - s1) < dist_from_center * 2:
        raise PathSplitError("Fist segment of a chunk is too short to put 3 extra points there")
    return waypoint_between_in_distance(middle, s0, dist_from_center), middle, \
        waypoint_between_in_distance(middle, s1, dist_from_center)


def move_connections_to_segments(max_speed, max_acc, split_chunks: List[Tuple[float, np.array]]) -> List[PathChunk]:
    res = []
    for i in range(len(split_chunks)):
        rotation, path = split_chunks[i]
        if i != 0:
            first_cut = _cut_segment(path[0], path[1])
        else:
            first_cut = path[0], path[0], path[0]

        if i != len(split_chunks) - 1:
            _, next_path = split_chunks[i + 1]
            last_cut = _cut_segment(next_path[0], next_path[1])
        else:
            last_cut = path[-1], path[-1], path[-1]

        in_speed = min(max_speed, _get_max_final_speed(np.linalg.norm(path[0] - first_cut[1]), max_acc))
        out_speed = min(max_speed, _get_max_final_speed(np.linalg.norm(path[-1] - last_cut[1]), max_acc))
        new_waypoints = np.concatenate(([first_cut[1], first_cut[2]], path[1:], [last_cut[0], last_cut[1]]))
        res.append(PathChunk(in_speed, out_speed, rotation, new_waypoints))
    return res


def split_waypoints_into_sweeping_chunks(way_pts) -> List[Tuple[float, np.array]]:
    """
    Split the waypoints into chunks of sweeping in one direction
    :param way_pts: original waypoints
    :return: list of chunks in form (rotation_angle, way_points),
    where rotation_angle is the angle of sweeping direction in radians: 0 for vertical, pi / 2 for horizontal
    """
    res = []
    cur_chunk_start = 0
    num_violated = 0  # Constantly updated number of segments with different rotation than
    cur_chunk_rotation = _segment_rotation(way_pts[0], way_pts[1])

    i = 0
    while i < len(way_pts) - 1:
        cur_angle = _segment_rotation(way_pts[i], way_pts[i + 1])
        if abs(cur_angle - cur_chunk_rotation) < ROTATION_VIOLATION_EPS:
            num_violated = 0
            i += 1
            continue
        if num_violated == 0:
            num_violated += 1
            i += 1
            continue

        # If it's already second violated segment, consider that the chunk is done and a new one is started
        res.append((cur_chunk_rotation, way_pts[cur_chunk_start:i]))
        cur_chunk_start = i - 1
        num_violated = 0
        cur_chunk_rotation = _segment_rotation(way_pts[i - 1], way_pts[i])

    res.append((cur_chunk_rotation, way_pts[cur_chunk_start:]))
    return res


def rotate_path(way_pts, angle: float) -> np.array:
    """
    NOTE: this function modifies way_pts!!! and does not return anything
    :param way_pts: waypoints to be rotated
    :param angle:angle by which the rotation should be performed
    """
    new_p = np.array(way_pts, copy=True, dtype='float64')
    # new_p = np.zeros(way_pts.shape)
    for i in range(way_pts.shape[0]):
        x, y = float(way_pts[i][0]), float(way_pts[i][1])
        new_p[i][0] = x * cos(angle) - y * sin(angle)
        new_p[i][1] = x * sin(angle) + y * cos(angle)
    return new_p


def _equidistant_points_in_segment(w1: np.array, w2: np.array, distance: float) -> np.array:
    d = np.linalg.norm(w2 - w1)
    num_of_added = ceil(d / distance)
    res = [w1]
    for i in range(1, num_of_added):
        res.append(waypoint_between_in_distance(w1, w2, d * (i / num_of_added)))
    return np.array(res)


def add_waypoints_with_distance(init_way_pts: np.array, distance: float):
    res = []
    for i in range(init_way_pts.shape[0] - 1):
        res.append(_equidistant_points_in_segment(init_way_pts[i], init_way_pts[i + 1], distance))
    res.append(np.array([init_way_pts[-1]]))
    return np.concatenate(res)

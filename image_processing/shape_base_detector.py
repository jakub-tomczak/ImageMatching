from utils.dataset_helper import Image
from utils.debug_conf import *
from utils.points_helpers import take_two_subseqent_points_indices, distance
from utils.model import Angle


def find_base_of_shape(coords: [[float, float]], distances: [[float, int]]) -> [int, int]:
    """returns line that is the most probable base of the shape
    returning line is a tuple of two subsequent coords or None
    TODO
    take two subsequent points and check wheter they create a line (example set7/1.png)
    """
    base_of_shape_line = None
    right_angle_detection_margin = 10

    n = max(4, int(len(distances) * 0.4))  # allow top n distances to be taken into consideration
    end_iteration = min(n, len(distances))
    if end_iteration < 1 and DEBUG and DEBUG_FIND_BASE:
        print("No distances!")

    for i in range(end_iteration):
        if DEBUG and DEBUG_FIND_BASE:
            print('finding base, iter = {}/{}'.format(i, end_iteration - 1))

        candidate_start_index, candidate_end_index = \
            take_two_subseqent_points_indices(distances[i][1], len(coords), True)
        candidate_start = coords[candidate_start_index]
        candidate_end = coords[candidate_end_index]

        # point before candidate_start
        previous_point_index, _ = take_two_subseqent_points_indices(candidate_start_index, len(coords), False)
        previous_point = coords[previous_point_index]
        # point after candidate_end
        _, next_point_index = take_two_subseqent_points_indices(candidate_end_index, len(coords), True)
        next_point = coords[next_point_index]

        if DEBUG and DEBUG_FIND_BASE:
            print('checking points {} {} {} and {} {} {}'.format(previous_point, candidate_start, candidate_end,
                                                                 candidate_start, candidate_end, next_point))
        angle_between_prev_and_start = Angle.calculate_angle_between(previous_point, candidate_start, candidate_end)
        previous_point_angle = abs(angle_between_prev_and_start - 90)
        angle_between_start_and_end = Angle.calculate_angle_between(candidate_start, candidate_end, next_point)
        next_point_angle = abs(angle_between_start_and_end - 90)

        if DEBUG and DEBUG_FIND_BASE:
            print('result is {} ({}) and {} ({})'.format(
                previous_point_angle, angle_between_prev_and_start, next_point_angle, angle_between_start_and_end)
            )

        if previous_point_angle < right_angle_detection_margin and next_point_angle < right_angle_detection_margin:
            base_of_shape_line = (candidate_start_index, candidate_end_index)
            if DEBUG and DEBUG_FIND_BASE:
                print("OK")
            break
    return base_of_shape_line


def find_best_bases(angles: [Angle]):
    return [i for i in sorted(enumerate(angles), key=lambda x: x[1].armA.length, reverse=True)][:2]


def find_base(image: Image, ang: [float]):
    # on 0th position we store distance between
    # 0th coord and 1st coord
    distances = []
    coords_num = len(image.points_coords) - 1  # the last coord is equal to the first one so skip it
    # calculated distances between points and append to a list
    if coords_num > 1:
        for i in range(coords_num):
            p0, p1 = take_two_subseqent_points_indices(i, len(image.points_coords), True)
            distances.append((distance(image.points_coords[p0], image.points_coords[p1]), i))

    distances.sort(key=lambda x: x[0], reverse=True)
    best_candidate_for_base = find_base_of_shape(image.points_coords, distances)

    return find_best_bases(ang), distances
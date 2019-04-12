from skimage.measure import find_contours, approximate_polygon
from skimage.transform import resize
import numpy as np
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import *
from utils.debug_helper import show_comparing_points, show_debug_info, draw_image_spec
from utils.model import ImageAngleData, Angle, Arm
from utils.mutators import compress_points
from utils.points_helpers import distance

NO_SKIP_POSSIBLE = 3


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = []
        shorter, longer, first_as_first = (first, second, True) \
            if len(first.angles) < len(second.angles) \
            else (second, first, False)
        shorter_len = len(shorter.angles)
        longer_len = len(longer.angles)
        show = False
        for offset_a, ang1 in shorter.possible_bases:
            for offset_b, ang2 in longer.possible_bases:
                a_i = 0
                b_i = 0
                values = []
                points = []
                while a_i < shorter_len and b_i < longer_len:
                    ap, bp, sim = CompareResult.find_matching_angle(shorter.angles, longer.angles, a_i, b_i, offset_a,
                                                                    offset_b)
                    points.append((
                        shorter.angles[(a_i + ap + offset_a) % shorter_len],
                        longer.angles[(offset_b - b_i - bp) % longer_len])
                    )
                    a_i += ap + 1
                    b_i += bp + 1
                    values.append(sim)
                if show:
                    show_comparing_points(first.image.data, second.image.data, points, first_as_first)
                different_offsets.append(sum(values) / max((shorter_len - 4), 1))  # -2 because of the base
        self.similarity = max(different_offsets)

    @staticmethod
    def find_matching_angle(a, b, a_i, b_i, a_offset, b_offset):
        len_a = len(a)
        len_b = len(b)
        for a_o in range(NO_SKIP_POSSIBLE):
            a_angle = a[(a_i + a_o + a_offset) % len_a]
            for b_o in range(NO_SKIP_POSSIBLE):
                b_angle = b[(b_offset - b_i - b_o) % len_b]
                sim = a_angle.mirror_similarity(b_angle) if a_angle.can_match(b_angle) else 0
                if sim > 0:
                    return a_o, b_o, sim
        return NO_SKIP_POSSIBLE - 1, NO_SKIP_POSSIBLE - 1, 0


def calculate_meaningful_points(coords: [[int, int]], min_distance: float):
    coords = compress_points(coords, min_distance)
    arms = calculate_arms(coords)
    arms = merge_half_full_arms(arms)
    ang = calculate_angles_for_arms(arms)
    return arms, ang


def calculate_angles_for_arms(arms: [Arm]):
    last = arms[len(arms) - 1]
    angles = []
    for a in arms:
        angles.append(Angle(last, a))
        last = a
    return angles


def merge_half_full_arms(arms: [Arm]):
    merged_arms = []
    last = arms[len(arms) - 1]
    was_added = False
    for a in arms:
        angle = Angle.for_points(last.a, a.a, a.b)
        was_added = not angle.is_half_full()
        if was_added:
            merged_arms.append(last)
            last = a
        else:
            last = Arm(last.a, a.b)
    if not was_added:
        if len(merged_arms) > 0:
            previous_first = merged_arms.pop(0)
            last = Arm(last.a, previous_first.b)
        merged_arms.append(last)
    return merged_arms


def calculate_arms(coords: [[int, int]]):
    arms = []
    last_coord = coords[len(coords) - 1]
    for c in coords:
        arms.append(Arm(last_coord, c))
        last_coord = c
    return arms


# takes returns tuple (point, other_point)
# other point is the next one if is_other_next_point
# other point is the previous one if not is_other_next_point
def take_two_subseqent_points_indices(point_index: int, number_of_points: int, is_other_next_point: bool) -> (int, int):
    if is_other_next_point:
        return point_index, (point_index + 1) % (number_of_points - 1)
    else:
        return (point_index - 1) % (number_of_points - 1), point_index


# returns line that is the most probable base of the shape
# returning line is a tuple of two subsequent coords or None
# TODO
# take two subsequent points and check wheter they create a line (example set7/1.png)
def find_base_of_shape(coords: [[float, float]], distances: [[float, int]]) -> [int, int]:
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


def angles(img: Image):
    if DEBUG:
        print("{}image_{}{}".format('\n' * 2, img.name, '-' * 20))
    image = img.data
    image = resize(image, (image.shape[0] * 4, image.shape[1] * 4), anti_aliasing=True)
    con = find_contours(image, .8)
    contour = con[0]
    min_distance = (image.shape[0] + image.shape[1]) / 100
    coords = approximate_polygon(contour, tolerance=min_distance / 2)
    arms, ang = calculate_meaningful_points(coords[:-1], min_distance)

    # on 0th position we store distance between
    # 0th coord and 1st coord
    distances = []
    coords_num = len(coords) - 1  # the last coord is equal to the first one so skip it
    # calculated distances between points and append to a list
    if coords_num > 1:
        for i in range(coords_num):
            p0, p1 = take_two_subseqent_points_indices(i, len(coords), True)
            distances.append((distance(coords[p0], coords[p1]), i))

    distances.sort(key=lambda x: x[0], reverse=True)
    best_candidate_for_base = find_base_of_shape(coords, distances)

    best_bases = find_best_bases(ang)

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, arms, coords, image, distances, best_candidate_for_base)

    return ang, best_bases


def get_ranking(dataset: Dataset):
    ang = [prepare_image_data(img) for img in dataset.images]
    for i1, a1 in enumerate(ang):
        for i2, a2 in enumerate(ang[i1 + 1:]):
            compare_res = CompareResult(a1, a2)
            a1.comparisons[a2] = compare_res
            a2.comparisons[a1] = compare_res

    return [[r[0].image.name for r in a.ranking()] for a in ang]


def prepare_image_data(img: Image):
    ang, bases = angles(img)
    return ImageAngleData(img, ang, bases)


def find_best_bases(angles: [Angle]):
    return [i for i in sorted(enumerate(angles), key=lambda x: x[1].armA.length, reverse=True)][:3]


if __name__ == '__main__':
    coords = np.array([
        [157, 5], [5, 5], [5, 13], [8, 100], [5, 140],
        [5, 150], [100, 150], [150, 150], [150, 140], [150, 100], [150, 50]
    ])
    image = np.zeros((160, 160))
    from matplotlib import pyplot as plt

    arms, ang = calculate_meaningful_points(coords, 1)
    draw_image_spec(image, ang,arms, coords)
    plt.show()

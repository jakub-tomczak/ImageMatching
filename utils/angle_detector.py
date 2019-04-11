from skimage.measure import find_contours, approximate_polygon
from skimage.transform import resize

from utils.dataset_helper import Image, Dataset
from utils.debug_conf import *
from utils.debug_helper import show_comparing_points, show_debug_info
from utils.model import ImageAngleData, Angle
from utils.mutators import compress_points
from utils.points_helpers import distance, accumulate_points

NO_SKIP_POSSIBLE = 3
ACCEPT_STRAIGHT_ANGLE_DIF = 15
MIN_DISTANCE = 20


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


def calculate_angle_for_point_at(points, it: int, points_number: int):
    return Angle(
        points[(it - 1) % points_number],
        points[it],
        points[(it + 1) % points_number]
    )


def compute_angles(coords):
    coords = compress_points(coords, MIN_DISTANCE)
    points_num = len(coords)
    valid_angles = []
    start = None
    points = []
    for i in range(points_num):
        a = calculate_angle_for_point_at(coords, i, points_num)
        if start is not None:
            points.append(a.point)
            center = accumulate_points(points)
            a = Angle(start, center, a.armB.b)
        if abs(180 - a.angle) > ACCEPT_STRAIGHT_ANGLE_DIF:
            start = None
            points = []
            valid_angles.append(a)
        else:
            if start is None:
                start = a.armA.a
                points.append(a.point)
    if start is not None:
        center = accumulate_points(points)
        a = Angle(start, center, coords[0])
        valid_angles.append(a)

    return valid_angles


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

        candidate_start_index, candidate_end_index = take_two_subseqent_points_indices(distances[i][1], len(coords),
                                                                                       True)
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
    coords = approximate_polygon(contour, tolerance=6)
    ang = compute_angles(coords[:-1])

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

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, coords, image, distances, best_candidate_for_base)

    return ang


def get_ranking(dataset: Dataset):
    ang = [ImageAngleData(img, angles(img)) for img in dataset.images]
    for i1, a1 in enumerate(ang):
        for i2, a2 in enumerate(ang[i1 + 1:]):
            compare_res = CompareResult(a1, a2)
            a1.comparisons[a2] = compare_res
            a2.comparisons[a1] = compare_res

    return [[r[0].image.name for r in a.ranking()] for a in ang]

from skimage.measure import find_contours, approximate_polygon
from utils.debug_helper import show_angle_on_image
import numpy as np
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import *
from utils.debug_helper import show_comparing_points, show_debug_info, draw_image_spec
from utils.model import ImageAngleData, Angle, Arm, BaseArm, ComparePoint, ExtremeComparePoint, SerialComparePoint
from utils.mutators import compress_points
from utils.points_helpers import distance


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = []
        for f_angles in first.comparison_points:
            for s_angles in second.comparison_points:
                shorter, longer, first_as_first = (f_angles, s_angles, True) \
                    if len(f_angles) < len(s_angles) \
                    else (s_angles, f_angles, False)
                shorter_len = len(shorter)
                longer_len = len(longer)
                show = False
                a_i = 0
                b_i = longer_len - 1
                total_sim = 0
                points = []
                while a_i < shorter_len and b_i >= 0:
                    ap, bp, sim = CompareResult.find_matching_angle(shorter, longer, a_i, b_i)
                    if show and sim > 0:
                        points.append(
                            (shorter[min(a_i + ap, shorter_len - 1)], longer[max(b_i - bp, 0)]))
                    a_i += ap + 1
                    b_i -= bp + 1
                    total_sim += sim

                if show:
                    show_comparing_points(first.image.data, f_angles, second.image.data, s_angles, points,
                                          first_as_first)
                different_offsets.append(total_sim / max(longer_len, 1))
        self.similarity = max(different_offsets)

    @staticmethod
    def find_matching_angle(a: [ComparePoint], b: [ComparePoint], a_i: int, b_i: int):
        a_point, b_point, a_off, b_off = CompareResult.get_aligned_points(a, b, a_i, b_i)
        if a_point is None:
            return a_off, b_off, 0

        if a_point.can_compare_with(b_point):
            a0, b0, sim = a_point.similarity(b_point)
            return a0 + a_off, b0 + b_off, sim
        else:
            if b_i - b_off > 0:
                b_point2 = b[b_i - 1 - b_off]
                if a_point.can_compare_with(b_point2):
                    a0, b0, sim = a_point.similarity(b_point2)
                    return a0 + a_off, b0 + 1 + b_off, sim
            if a_i + a_off < len(a) - 1:
                a_point2 = a[a_i + 1 + a_off]
                if a_point2.can_compare_with(b_point):
                    a0, b0, sim = a_point2.similarity(b_point)
                    return a0 + 1 + a_off, b0 + b_off, sim
        return a_off, b_off, 0

    @staticmethod
    def get_aligned_points(a: [ComparePoint], b: [ComparePoint], a_i, b_i):
        def is_in_range():
            return a_i + a_off < len_a and b_i - b_off >= 0

        a_point: ComparePoint = a[a_i]
        b_point: ComparePoint = b[b_i]
        a_off = 0
        b_off = 0
        range_com = 0.1
        len_a = len(a)
        dif = a_point.progress_difference(b_point)

        while (dif > range_com or dif < -range_com) and is_in_range():
            if dif < -range_com:
                a_off += 1
                a_point = a[a_i + a_off]
            elif dif > range_com:
                b_off += 1
                b_point: ComparePoint = b[b_i - b_off]
            dif = a_point.progress_difference(b_point)
        if not is_in_range():
            return None, None, a_off, b_off
        return a_point, b_point, a_off, b_off


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


def angles(img: Image):
    image = img.data
    # image = resize(image, (image.shape[0] * 4, image.shape[1] * 4), anti_aliasing=True)
    # img.data = image
    con = find_contours(image, .8)
    contour = max(con, key=lambda x: len(x))
    min_distance = (image.shape[0] + image.shape[1]) / 100
    coords = approximate_polygon(contour, tolerance=min_distance / 0.75)
    arms, ang = calculate_meaningful_points(coords[:-1], min_distance * 3)

    arms_bases = find_best_bases(arms)

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, arms, coords, image, arms_bases)

    return ang, arms_bases


def get_ranking(dataset: Dataset):
    ang = [prepare_image_data(img) for img in dataset.images]
    for i1, a1 in enumerate(ang):
        for i2, a2 in enumerate(ang[i1 + 1:]):
            compare_res = CompareResult(a1, a2)
            a1.comparisons[a2] = compare_res
            a2.comparisons[a1] = compare_res
    if RESULT_DISPLAY_RANKING_POINTS:
        results = [[(r[0].image.name, r[1].similarity) for r in a.ranking()] for a in ang]
        for r in results:
            print(r)

    return [[r[0].image.name for r in a.ranking()] for a in ang]


def prepare_image_data(img: Image):
    ang, bases = angles(img)
    angles_count = len(ang)
    comparison_angles = []
    max_angles_dist = (img.data.shape[0] + img.data.shape[1]) / 40
    for b in bases:
        eng_angle_i = (b.end + 2) % angles_count
        if b.start < eng_angle_i:
            new_ang = ang[eng_angle_i:] + ang[:b.start]
        else:
            new_ang = ang[eng_angle_i:b.start]
        first = ExtremeComparePoint(new_ang[0], True)
        last_point_num = len(new_ang) - 1
        last = ExtremeComparePoint(new_ang[last_point_num], False)
        serials = [first]
        for i, a in enumerate(new_ang):
            if i == 0 or i == last_point_num:
                continue
            bef, aft = neighbours(a, i, new_ang, max_angles_dist)
            location_angle = Angle.for_points(first.angle.point, a.point, last.angle.point)
            serials.append(SerialComparePoint(a, location_angle, bef, aft))
        serials.append(last)
        comparison_angles.append(serials)
    return ImageAngleData(img, comparison_angles)


def neighbours(ang: Angle, index: int, all: [Angle], max_distance: float):
    before = []
    after = []
    for i in range(index - 1, 0, -1):
        bef = all[i]
        dist = distance(ang.point, bef.point)
        if dist <= max_distance or len(before) < 1:
            before.append(bef)
        else:
            break
    for i in range(index + 1, len(all) - 1):
        aft = all[i]
        dist = distance(ang.point, aft.point)
        if dist <= max_distance or len(after) < 1:
            after.append(aft)
        else:
            break
    return before, after


def find_best_bases(arms: [Arm]):
    max_search = min(3, int(len(arms) * 0.2))

    def is_close_to_straight_angle(arm1: Arm, arm2: Arm):
        ang = Angle.calculate_angle_between(arm1.a, arm2.a, arm2.b)
        return abs(270 - ang) < 20

    def search_further(start: int, i: int, current: Arm):
        next_arm = arms[i % total]
        if is_close_to_straight_angle(current, next_arm):
            return i, current
        elif abs(start - i) < max_search and next_arm.length / current.length < 0.2:
            return search_further(start, i + 1, Arm(current.a, next_arm.b))
        return i, None

    def search_previous(start: int, i: int, current: Arm):
        previous_arm = arms[i % total]
        if is_close_to_straight_angle(previous_arm, current):
            return i, current
        elif abs(start - i) < max_search and previous_arm.length / current.length < 0.2:
            return search_previous(start, i - 1, Arm(previous_arm.a, current.b))
        return i, None

    candidates = [i for i in sorted(enumerate(arms), key=lambda x: x[1].length, reverse=True)][:4]
    result = []
    total = len(arms)
    for i, c in candidates:
        end_i, extended = search_further(i, i + 1, c)
        if extended is None:
            continue
        start_i, extended = search_previous(i, i - 1, extended)
        if extended is None:
            continue
        if len(result) > 0 and extended.length / result[0].arm.length < 0.75:
            continue
        result.append(BaseArm((start_i + 1) % total, (end_i - 1) % total, extended))

    if len(result) == 0:
        result = [BaseArm(c[0], c[0], c[1]) for c in candidates[:3]]
    return result


if __name__ == '__main__':
    coords = np.array([
        [157, 5], [5, 5], [5, 13], [8, 100], [5, 140],
        [5, 150], [100, 150], [150, 150], [150, 140], [150, 100], [150, 50]
    ])
    image = np.zeros((160, 160))
    from matplotlib import pyplot as plt

    arms, ang = calculate_meaningful_points(coords, 1)
    draw_image_spec(image, ang, arms, coords)
    plt.show()

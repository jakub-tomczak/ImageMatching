from skimage.measure import find_contours, approximate_polygon
from skimage.transform import resize

from image_processing.shape_base_detector import find_base
import numpy as np
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import *
from utils.debug_helper import show_comparing_points, show_debug_info, draw_image_spec
from utils.model import ImageAngleData, Angle, Arm, BaseArm
from utils.mutators import compress_points

NO_SKIP_POSSIBLE = 3


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = []
        for f_angles in first.comparison_angles:
            for s_angles in second.comparison_angles:
                shorter, longer, first_as_first = (f_angles, s_angles, True) \
                    if len(f_angles) < len(s_angles) \
                    else (s_angles, f_angles, False)
                shorter_len = len(shorter)
                longer_len = len(longer)
                show = False
                a_i = 0
                b_i = 0
                values = []
                points = []
                while a_i < shorter_len and b_i < longer_len:
                    ap, bp, sim = CompareResult.find_matching_angle(shorter, longer, a_i, b_i)
                    points.append((shorter[min(a_i + ap, shorter_len - 1)], longer[max(longer_len - 1 - b_i - bp, 0)]))
                    a_i += ap + 1
                    b_i += bp + 1
                    values.append(sim)
                if show:
                    show_comparing_points(first.image.data, f_angles, second.image.data, s_angles, points,
                                          first_as_first)
                different_offsets.append(sum(values) / max(shorter_len, 1))
        self.similarity = max(different_offsets)

    @staticmethod
    def find_matching_angle(a: [Angle], b: [Angle], a_i: int, b_i: int):
        len_a = len(a)
        len_b = len(b)
        a_first = a[a_i]
        b_first = b[len_b - b_i - 1]
        for a_o in range(NO_SKIP_POSSIBLE):
            a_next_index = a_i + a_o
            if a_next_index >= len_a:
                break
            a_angle = a[a_next_index]
            if a_o > 0:
                a_angle = Angle.for_points(a_first.armA.a, a_angle.armB.a, a_angle.armB.b)
            for b_o in range(NO_SKIP_POSSIBLE):
                b_next_index = len_b - b_i - b_o - 1
                if b_next_index < 0:
                    break
                b_angle = b[b_next_index]
                if b_o > 0:
                    b_angle = Angle.for_points(b_angle.armA.a, b_angle.armA.b, b_first.armB.b)
                first_or_last = (b_next_index == 0 or b_next_index == len_b - 1) and \
                                (a_next_index == 0 or a_next_index == len_a - 1)
                sim = a_angle.mirror_similarity(b_angle, first_or_last) if first_or_last or a_angle.can_match(
                    b_angle) else 0
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


def angles(img: Image, resize_image: bool = True):
    if resize_image:
        img.data = resize(img.data, (img.data.shape[0] * 4, img.data.shape[1] * 4), anti_aliasing=True)
    con = find_contours(img.data, .8)
    contour = max(con, key=lambda x: len(x))
    min_distance = (img.data.shape[0] + img.data.shape[1]) / 100
    coords = approximate_polygon(contour, tolerance=min_distance / 2)
    img.points_coords = coords[:-1]
    arms, ang = calculate_meaningful_points(coords[:-1], min_distance)

    arms_bases = find_best_bases(arms)
    img.arms = arms

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, arms, coords, img.data, arms_bases)

    return ang, arms_bases


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
    angles_count = len(ang)
    comparison_angles = []
    for b in bases:
        eng_angle_i = (b.end + 2) % angles_count
        if b.start < eng_angle_i:
            new_ang = ang[eng_angle_i:] + ang[:b.start]
        else:
            new_ang = ang[eng_angle_i:b.start]
        comparison_angles.append(new_ang)
    return ImageAngleData(img, comparison_angles)


def find_best_bases(arms: [Arm]):
    max_search = min(3, int(len(arms) * 0.2))

    def is_close_to_right_angle(arm1: Arm, arm2: Arm):
        ang = Angle.calculate_angle_between(arm1.a, arm2.a, arm2.b)
        return abs(270 - ang) < 20

    def search_further(start: int, i: int, current: Arm):
        next_arm = arms[i % total]
        if is_close_to_right_angle(current, next_arm):
            return i, current
        elif abs(start - i) < max_search and current.length > 0 and next_arm.length / current.length < 0.2:
            return search_further(start, i + 1, Arm(current.a, next_arm.b))
        return i, None

    def search_previous(start: int, i: int, current: Arm):
        previous_arm = arms[i % total]
        if is_close_to_right_angle(previous_arm, current):
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

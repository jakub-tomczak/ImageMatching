import math
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, approximate_polygon
from skimage.transform import rotate

from utils.compare_result import CompareResult
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import *
from utils.debug_helper import show_debug_info, draw_image_spec
from utils.model import ImageAngleData, Angle, Arm, BaseArm, ExtremeComparePoint, SerialComparePoint
from utils.mutators import compress_points
from utils.points_helpers import distance


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
    local_thresh = threshold_otsu(image)
    image = image > local_thresh
    img.data = image
    con = find_contours(image, .8)
    contour = max(con, key=lambda x: len(x))
    min_distance = (image.shape[0] + image.shape[1]) / 100
    coords = approximate_polygon(contour, tolerance=min_distance / 0.75)
    arms, ang = calculate_meaningful_points(coords[:-1], min_distance * 3)

    arms_bases = find_best_bases(arms)
    slope_deg = arms_bases[0].arm.slope_angle()
    if DEBUG:
        image = rotate(image, slope_deg)
        img.data = image
    slope = math.radians(slope_deg)
    origin = (image.shape[0] / 2, image.shape[1] / 2)
    for a in arms:
        a.a = rotate_point(origin, a.a, slope)
        a.b = rotate_point(origin, a.b, slope)
    for a in ang:
        a.point = rotate_point(origin, a.point, slope)
        a.point_a = rotate_point(origin, a.point_a, slope)
        a.point_c = rotate_point(origin, a.point_c, slope)

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, arms, coords, image, arms_bases)

    return ang, arms_bases


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


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

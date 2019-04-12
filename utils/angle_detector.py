from skimage.measure import find_contours, approximate_polygon
from skimage.transform import resize

from image_processing.shape_base_detector import find_base
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import *
from utils.debug_helper import show_comparing_points, show_debug_info
from utils.model import ImageAngleData, Angle
from utils.mutators import compress_points
from utils.points_helpers import distance, accumulate_points, take_two_subseqent_points_indices

NO_SKIP_POSSIBLE = 3
ACCEPT_STRAIGHT_ANGLE_DIF = 10


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


def compute_angles(coords, min_distance):
    coords = compress_points(coords, min_distance)
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


def angles(img: Image):
    if DEBUG:
        print("{}image_{}{}".format('\n' * 2, img.name, '-' * 20))
    image = img.data
    image = resize(image, (image.shape[0] * 4, image.shape[1] * 4), anti_aliasing=True)
    con = find_contours(image, .8)
    contour = con[0]
    min_distance = (image.shape[0] + image.shape[1]) / 100
    img.points_coords = approximate_polygon(contour, tolerance=min_distance / 2)
    ang = compute_angles(img.points_coords[:-1], min_distance)
    return ang


def get_ranking(dataset: Dataset):
    ang = [prepare_image_data(img) for img in dataset.images]
    for i1, a1 in enumerate(ang):
        for i2, a2 in enumerate(ang[i1 + 1:]):
            compare_res = CompareResult(a1, a2)
            a1.comparisons[a2] = compare_res
            a2.comparisons[a1] = compare_res

    return [[r[0].image.name for r in a.ranking()] for a in ang]


def prepare_image_data(img: Image):
    ang = angles(img)
    bases, distances = find_base(img, ang)

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, img.points_coords, img, distances, bases[0])

    return ImageAngleData(img, ang, bases)

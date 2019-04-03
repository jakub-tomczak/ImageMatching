import math

from matplotlib import pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from skimage.transform import resize

from utils.dataset_helper import Image, Dataset

DEBUG = False
NO_SKIP_POSSIBLE = 3
ACCEPT_STRAIGHT_ANGLE_DIF = 10


class Angle:
    def __init__(self, a, b, c) -> None:
        self.armA = Angle.pitagoras(c[1] - b[1], c[0] - b[0])
        self.armB = Angle.pitagoras(a[1] - b[1], a[0] - b[0])
        self.angle = Angle.calculate_angle_between(a, b, c)
        self.point = b

    @staticmethod
    def calculate_angle_between(a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    @staticmethod
    def pitagoras(a, b):
        return pow(a ** 2 + b ** 2, 0.5)

    def __repr__(self) -> str:
        return "{} ({}, {})".format(self.angle, self.armA, self.armB)

    def can_match(self, other):
        return abs(self.armB / other.armB - self.armA / other.armA) < 0.5 or \
               abs(self.armA / other.armB - self.armB / other.armA) < 0.5

    def mirror_similarity(self, other):
        return 1 - abs((self.angle + other.angle) / 360 - 1)


class ImageAngleData:
    def __init__(self, image: Image, angles: list):
        self.image = image
        self.angles = [a for a in angles if abs(180 - a.angle) > ACCEPT_STRAIGHT_ANGLE_DIF]
        self.angles_to_compare = self.angles[:: -1]
        self.comparisons = dict()

    def ranking(self):
        return sorted(self.comparisons.items(), key=lambda x: x[1].similarity, reverse=True)


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = dict()
        shorter, longer, first_as_first = (first.angles, second.angles_to_compare, True) if len(first.angles) < len(
            second.angles) else (
            second.angles, first.angles_to_compare, False)
        shorter_len = len(shorter)
        longer_len = len(longer)
        show = False
        # if first.image.name == 0 and second.image.name == 6:
        #     show = True
        for offset in range(shorter_len):
            a_i = 0
            b_i = 0
            values = []
            points = []
            while a_i < shorter_len and b_i < longer_len:
                ap, bp, sim = CompareResult.find_matching_angle(shorter, longer, a_i, b_i, offset)
                points.append((shorter[(a_i + ap) % shorter_len], longer[(b_i + bp + offset) % longer_len]))
                a_i += ap + 1
                b_i += bp + 1
                values.append(sim)
            if show:
                show_points(first.image.data, second.image.data, points, first_as_first)
            different_offsets[offset] = sum(values) / max((shorter_len - 2), 1)  # -2 because of the base
        self.similarity = max(different_offsets.values())

    @staticmethod
    def find_matching_angle(a, b, a_i, b_i, b_offset):
        len_a = len(a)
        len_b = len(b)
        for a_o in range(NO_SKIP_POSSIBLE):
            a_angle = a[(a_i + a_o) % len_a]
            for b_o in range(NO_SKIP_POSSIBLE):
                b_angle = b[(b_i + b_offset) % len_b]
                sim = a_angle.mirror_similarity(b_angle) if a_angle.can_match(b_angle) else 0
                if sim > 0:
                    return a_o, b_o, sim
        return 0, 0, 0


def calculate_angle_for_point_at(points, it: int, points_number: int):
    return Angle(
        points[(it + points_number - 1) % points_number],
        points[it],
        points[(it + points_number + 1) % points_number]
    )


def compute_angles(coords):
    points_num = len(coords)
    return [calculate_angle_for_point_at(coords, i, points_num) for i in range(points_num)]


def angles(img: Image):
    image = img.data
    image = resize(image, (image.shape[0] * 4, image.shape[1] * 4), anti_aliasing=True)
    con = find_contours(image, .8)
    contour = con[0]
    coords = approximate_polygon(contour, tolerance=6)
    ang = compute_angles(coords[:-1])
    if DEBUG:
        show_debug_info(ang, coords, image)
    return ang


def show_debug_info(ang, coords, image):
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r)
    ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=3)
    ax.plot(coords[0, 1], coords[0, 0], '*', color='blue')
    ax.plot(coords[1, 1], coords[1, 0], '*', color='green')
    ax.plot(coords[-2, 1], coords[-2, 0], 'o', color='orange')
    print(ang)
    plt.show()


def show_points(img1, img2, points, first_as_first):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(img1, interpolation='nearest', cmap=plt.cm.Greys_r)
    ax2.imshow(img2, interpolation='nearest', cmap=plt.cm.Greys_r)
    for i, (f1, f2) in enumerate(points):
        add_points(ax1, ax2, f1, f2, first_as_first, i)
    plt.show()


def add_points(ax1, ax2, p1, p2, is_first_first, index):
    color = ('blue', 'green')
    if index == 0:
        color = ('red', 'red')
    elif index == 1:
        color = ('orange', 'orange')
    pp1, pp2 = (p1.point, p2.point) if is_first_first else (p2.point, p1.point)
    ax1.plot(pp1[1] / 4, pp1[0] / 4, '*', color=color[0])
    ax2.plot(pp2[1] / 4, pp2[0] / 4, '*', color=color[1])


def get_ranking(dataset: Dataset):
    ang = [ImageAngleData(img, angles(img)) for img in dataset.images]
    for i1, a1 in enumerate(ang):
        for i2, a2 in enumerate(ang[i1 + 1:]):
            compare_res = CompareResult(a1, a2)
            a1.comparisons[a2] = compare_res
            a2.comparisons[a1] = compare_res

    return [[r[0].image.name for r in a.ranking()] for a in ang]

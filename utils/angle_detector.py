import math

from matplotlib import pyplot as plt
from skimage.measure import find_contours, approximate_polygon

from utils.dataset_helper import Image, Dataset


class Angle:
    def __init__(self, a, b, c) -> None:
        self.armA = Angle.pitagoras(c[1] - b[1], c[0] - b[0])
        self.armB = Angle.pitagoras(a[1] - b[1], a[0] - b[0])
        self.angle = Angle.calculate_angle_between(a, b, c)

    @staticmethod
    def calculate_angle_between(a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return ang + 360 if ang < 0 else ang

    @staticmethod
    def pitagoras(a, b):
        return pow(a ** 2 + b ** 2, 0.5)

    def __repr__(self) -> str:
        return "{} ({}, {})".format(self.angle, self.armA, self.armB)

    def can_compare_with(self, other):
        return abs(self.armB / other.armB - self.armA / other.armA) < 0.5 or \
               abs(self.armA / other.armB - self.armB / other.armA) < 0.5

    def mirror_similarity(self, other):
        return 1 - pow((self.angle + other.angle) / 360 - 1, 2)


class ImageAngleData:
    def __init__(self, image: Image, angles: list):
        self.image = image
        self.angles = angles
        self.angles_to_compare = self.angles[:: -1]
        self.comparisons = dict()

    def ranking(self):
        return sorted(self.comparisons.items(), key=lambda x: x[1].similarity, reverse=True)


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = dict()
        shorter, longer = (first.angles, second.angles_to_compare) if len(first.angles) < len(second.angles) else (second.angles, first.angles_to_compare)
        shorter_len = len(shorter)
        longer_len = len(longer)
        for offset in range(shorter_len):
            values = [a.mirror_similarity(longer[(offset + i) % longer_len])
                      for i, a in enumerate(shorter)
                      if a.can_compare_with(longer[(offset + i) % longer_len])]
            different_offsets[offset] = sum(values) / (shorter_len - 2)  # -2 because of the base
        self.similarity = max(different_offsets.values())


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
    # image = gaussian(image, sigma=0.5)
    con = find_contours(image, .799)
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r)

    contour = con[0]
    coords = approximate_polygon(contour, tolerance=2.5)
    ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=3)
    ax.plot(coords[0, 1], coords[0, 0], '*', color='blue')
    ax.plot(coords[1, 1], coords[1, 0], '*', color='green')
    ax.plot(coords[-2, 1], coords[-2, 0], 'o', color='orange')
    ang = compute_angles(coords[:-1])
    print(ang)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return ang


def get_ranking(dataset: Dataset):
    ang = [ImageAngleData(img, angles(img)) for img in dataset.images]
    for i1, a1 in enumerate(ang):
        for i2, a2 in enumerate(ang[i1 + 1:]):
            compare_res = CompareResult(a1, a2)
            a1.comparisons[a2] = compare_res
            a2.comparisons[a1] = compare_res

    return [[r[0].image.name for r in a.ranking()] for a in ang]

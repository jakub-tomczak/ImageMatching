import math

from utils.dataset_helper import Image
from utils.points_helpers import distance


class Arm:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.length = distance(a, b)

    def angle(self):
        return math.degrees(math.atan2(self.a[0] - self.b[0], self.a[1] - self.b[1]))

    def __repr__(self) -> str:
        return "Arm({} - {} [{}])".format(self.a, self.b, self.length)


class Angle:
    def __init__(self, a, b, c) -> None:
        self.armA = Arm(a, b)
        self.armB = Arm(b, c)
        self.angle = Angle.calculate_angle_between(a, b, c)
        self.point = b

    @staticmethod
    def calculate_angle_between(a, b, c):
        ang = math.degrees(math.atan2(c[0] - b[0], c[1] - b[1]) - math.atan2(a[0] - b[0], a[1] - b[1]))
        return ang + 360 if ang < 0 else ang

    def __repr__(self) -> str:
        return "{} ({}, {})".format(self.angle, self.armA, self.armB)

    def can_match(self, other):
        first_ratio = self.armA.length / other.armB.length
        second_ratio = self.armB.length / other.armA.length
        return abs(1 - first_ratio / second_ratio) < 0.25

    def mirror_similarity(self, other):
        return 1 - abs((self.angle + other.angle) / 360 - 1)


class ImageAngleData:
    def __init__(self, image: Image, angles: list, possible_bases: [Arm]):
        self.image = image
        self.angles = angles
        self.possible_bases = possible_bases
        self.comparisons = dict()

    def ranking(self):
        return sorted(self.comparisons.items(), key=lambda x: x[1].similarity, reverse=True)

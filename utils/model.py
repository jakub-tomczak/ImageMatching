import math

from utils.dataset_helper import Image
from utils.points_helpers import distance

HALF_FULL_ANGLE_ACCEPT_THRESHOLD = 10


class Arm:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.length = distance(a, b)
        self._slope = None

    def slope_angle(self):
        if self._slope is None:
            self._slope = math.degrees(math.atan2(self.a[0] - self.b[0], self.a[1] - self.b[1]))
        return self._slope

    def __repr__(self) -> str:
        return "Arm({} - {} [{}])".format(self.a, self.b, self.length)


class BaseArm:
    def __init__(self, start: int, end: int, arm: Arm) -> None:
        self.start = start
        self.end = end
        self.arm = arm


class Angle:
    def __init__(self, a: Arm, b: Arm) -> None:
        self.armA = a
        self.armB = b
        self.angle = Angle.calculate_angle_between(a.a, b.a, b.b)
        self.point = b.a

    def is_half_full(self):
        return abs(180 - self.angle) <= HALF_FULL_ANGLE_ACCEPT_THRESHOLD

    @staticmethod
    def for_points(a, b, c):
        return Angle(Arm(a, b), Arm(b, c))

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

    def mirror_similarity(self, other, first_or_last=False):
        target = 540 if first_or_last else 360
        return 1 - abs((self.angle + other.angle) / target - 1)


class ImageAngleData:
    def __init__(self, image: Image, comparison_angles: [[Angle]]):
        self.image = image
        self.comparison_angles = comparison_angles
        self.comparisons = dict()

    def ranking(self):
        return sorted(self.comparisons.items(), key=lambda x: x[1].similarity, reverse=True)

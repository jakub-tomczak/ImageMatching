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
        self.point_a = a.a
        self.point_c = b.b

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
        if (first_ratio >= 1 > second_ratio) or first_ratio < 1 <= second_ratio:
            return False
        lower, higher = (first_ratio, second_ratio) if first_ratio < second_ratio \
            else (second_ratio, first_ratio)
        return abs(1 - lower / higher) < 0.3

    def mirror_similarity(self, other, first_or_last=False):
        target = 540 if first_or_last else 360
        return angles_complement(self.angle, other.angle, target)


def angles_complement(ang1: float, ang2: float, target: float):
    return pow(1 - abs((ang1 + ang2) / target - 1), 8)


class ComparePoint:
    def __init__(self, angle, progress) -> None:
        self.angle = angle
        self.progress = progress

    def progress_difference(self, other) -> float:
        return self.progress - (1 - other.progress)

    def can_compare_with(self, other) -> bool:
        return False

    def similarity(self, other) -> (int, int, float):
        return 0, 0, 0


class ExtremeComparePoint(ComparePoint):
    def __init__(self, angle: Angle, is_start: bool):
        super().__init__(angle, 0 if is_start else 1)

    def can_compare_with(self, other) -> bool:
        return isinstance(other, ExtremeComparePoint)

    def similarity(self, other) -> (int, int, float):
        return 0, 0, self.angle.mirror_similarity(other.angle, True)


class SerialComparePoint(ComparePoint):
    def __init__(self, angle: Angle, location_angle: Angle, neighbors_before: [Angle], neighbors_after: [Angle]):
        super().__init__(angle, location_angle.armA.length / (location_angle.armA.length + location_angle.armB.length))
        self.location_angle = location_angle
        self.__neighbors_before = neighbors_before
        self.__neighbors_after = neighbors_after
        self.__matchers_before = None
        self.__matchers_after = None

    def can_compare_with(self, other) -> bool:
        if not isinstance(other, SerialComparePoint):
            return False
        return self.location_angle.can_match(other.location_angle) and \
               self.location_angle.mirror_similarity(other.location_angle) > 0.25

    def similarity(self, other) -> (int, int, float):
        if self.angle.can_match(other.angle):
            return 0, 0, self.angle.mirror_similarity(other.angle, False)
        others = other._matchers_before()
        for i, o in others:
            if self.angle.can_match(o):
                return 0, i, self.angle.mirror_similarity(o)
        angles_after_this = self._matchers_after()
        for ia, a in angles_after_this:
            if a.can_match(other.angle):
                return ia, 0, a.mirror_similarity(other.angle)
            for io, o in others:
                if a.can_match(o):
                    return ia, io, a.mirror_similarity(o)
        return len(angles_after_this), len(others) - 1, 0

    def _matchers_before(self) -> [(int, Angle)]:
        if self.__matchers_before is None:
            self.__matchers_before = merge_angles_before(self.__neighbors_before, self.angle)
        return self.__matchers_before

    def _matchers_after(self) -> [(int, Angle)]:
        if self.__matchers_after is None:
            self.__matchers_after = merge_angles_after(self.angle, self.__neighbors_after)
        return self.__matchers_after


def merge_angles_after(before: Angle, after: [Angle]):
    result = []
    visited = [before.point]
    for ai, a in enumerate(after):
        visited.append(a.point)
        result.extend(make_angles_between(before.point_a, a.point_c, visited, ai + 1))
    return result


def merge_angles_before(before: [Angle], after: Angle):
    result = []
    visited = [after.point]
    for bi, b in enumerate(before):
        visited.append(b.point)
        result.extend(make_angles_between(b.point_a, after.point_c, visited, bi + 1))
    return result


def make_angles_between(start: [[float]], end: [[float]], centers: [[float]], offset):
    return [(offset, Angle.for_points(start, c, end)) for c in centers]


class ImageAngleData:
    def __init__(self, image: Image, comparison_points: [[SerialComparePoint]]):
        self.image = image
        self.comparison_points = comparison_points
        self.comparisons = dict()

    def ranking(self):
        return sorted(self.comparisons.items(), key=lambda x: x[1].similarity, reverse=True)

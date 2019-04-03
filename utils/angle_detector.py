import math

from matplotlib import pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from skimage.transform import resize

from utils.dataset_helper import Image, Dataset

DEBUG = False
DEBUG_FIND_BASE = True
DEBUG_DISPLAY_IMAGES = False
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


# takes returns tuple (point, other_point)
# other point is the next one if is_other_next_point
# other point is the previous one if not is_other_next_point
def take_two_subseqent_points_indices(point_index: int, number_of_points: int, is_other_next_point: bool) -> (int, int):
    if is_other_next_point:
        return point_index, (point_index + 1) % (number_of_points - 1)
    else:
        return (point_index- 1) % (number_of_points - 1), point_index

# returns line that is the most probable base of the shape
# returning line is a tuple of two subsequent coords or None
# TODO
# take two subsequent points and check wheter they create a line (example set7/1.png)
def find_base_of_shape(coords: [[float, float]], distances: [[float, int]]) -> [int, int]:
    base_of_shape_line = coords[distances[0][1]]
    right_angle_detection_margin = 10


    n = max(4, int(len(distances) * 0.4)) # allow top n distances to be taken into consideration
    end_iteration = min(n, len(distances))
    if end_iteration < 1 and DEBUG and DEBUG_FIND_BASE:
        print("No distances!")

    for i in range(end_iteration):
        if DEBUG and DEBUG_FIND_BASE:
            print('finding base, iter = {}/{}'.format(i, end_iteration-1))

        candidate_start_index, candidate_end_index = take_two_subseqent_points_indices(distances[i][1], len(coords), True)
        candidate_start = coords[candidate_start_index]
        candidate_end = coords[candidate_end_index]

        # point before candidate_start
        previous_point_index, _ = take_two_subseqent_points_indices(candidate_start_index, len(coords), False)
        previous_point = coords[previous_point_index]
        # point after candidate_end
        _, next_point_index = take_two_subseqent_points_indices(candidate_end_index, len(coords), True)
        next_point = coords[next_point_index]

        if DEBUG and DEBUG_FIND_BASE:
            print('checking points {} {} {} and {} {} {}'.format(previous_point, candidate_start, candidate_end, candidate_start, candidate_end, next_point))
        previous_point_angle = abs(Angle.calculate_angle_between(previous_point, candidate_start, candidate_end) - 90)
        next_point_angle = abs(Angle.calculate_angle_between(candidate_start, candidate_end, next_point) - 90)

        if DEBUG and DEBUG_FIND_BASE:
            print('result is {} ({}) and {} ({})'.format(previous_point_angle, Angle.calculate_angle_between(previous_point, candidate_start, candidate_end), \
                                                         next_point_angle,  Angle.calculate_angle_between(candidate_start, candidate_end, next_point)))

        if previous_point_angle < right_angle_detection_margin \
            and next_point_angle < right_angle_detection_margin:
            base_of_shape_line = (candidate_start_index, candidate_end_index)
            if DEBUG and DEBUG_FIND_BASE:
                print("OK")
            break
    return base_of_shape_line

def angles(img: Image):
    if DEBUG:
        print("{}image_{}{}".format('\n'*2, img.name, '-'*20))
    image = img.data
    image = resize(image, (image.shape[0] * 4, image.shape[1] * 4), anti_aliasing=True)
    con = find_contours(image, .8)
    contour = con[0]
    coords = approximate_polygon(contour, tolerance=6)
    ang = compute_angles(coords[:-1])

    # on 0th position we store distance between
    # 0th coord and 1st coord
    distances = []
    coords_num = len(coords) - 1 # the last coord is equal to the first one so skip it
    # calculated distances between points and append to a list
    if coords_num > 1:
        for i in range(coords_num):
            p0, p1 = take_two_subseqent_points_indices(i, len(coords), True)
            distances.append( (calculate_distance_between_points(coords[p0], coords[p1]), i) )

    distances.sort(key = lambda x: x[0], reverse=True)
    best_candidate_for_base = find_base_of_shape(coords, distances)

    if DEBUG and DEBUG_DISPLAY_IMAGES:
        show_debug_info(ang, coords, image, distances, best_candidate_for_base)

    return ang


def show_debug_info(ang, coords, image, distances, best_candidate_for_base):
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r)
    ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=3)
    ax.plot(coords[0, 1], coords[0, 0], '*', color='blue')
    ax.plot(coords[1, 1], coords[1, 0], '*', color='green')
    ax.plot(coords[-2, 1], coords[-2, 0], 'o', color='orange')

    # draw a few longest distances
    for i in range(1):
        color = 'yellow'
        if best_candidate_for_base != None:
            p_0_index, p_1_index = best_candidate_for_base
        else:
            color = 'blue'
            if i >= len(distances):
                break
            p_0_index = distances[i][1]
            p_1_index = (p_0_index + 1) % len(distances)

        xx = [coords[p_0_index][1], coords[p_1_index][1] ]
        yy = [coords[p_0_index][0], coords[p_1_index][0] ]
        ax.plot(xx, yy, 'ro-', color=color)

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

def calculate_distance_between_points(first_coords: (int, int), second_coords: (int, int)):
    return Angle.pitagoras(abs(first_coords[0] - second_coords[0]), abs(first_coords[1] - second_coords[1]))

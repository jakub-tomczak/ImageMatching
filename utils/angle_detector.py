import math

from matplotlib import pyplot as plt
from skimage.measure import find_contours, approximate_polygon

from utils.dataset_helper import Image, Dataset


class ImageAngleData:
    def __init__(self, image: Image, angles: list):
        self.image = image
        self.angles = angles
        self.angles_to_compare = self.angles[:: -1]
        self.comparisons = dict()

    def ranking(self):
        return sorted(self.comparisons.items(), key=lambda x: x[1].min_dif)


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = dict()
        other_len = len(second.angles_to_compare)
        for offset in range(len(first.angles)):
            pais = sum([pow((a + second.angles_to_compare[(offset + i) % other_len]) / 360 - 1, 2)
                        for i, a in enumerate(first.angles)])
            different_offsets[offset] = pais
        self.min_dif = min(different_offsets.values())


def angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def calculate_angle_for_point_at(points, it: int, points_number: int):
    return angle(
        points[(it + points_number - 1) % points_number],
        points[it],
        points[(it + 1 + points_number) % points_number]
    )


def compute_angles(coords):
    points_num = len(coords)
    return [calculate_angle_for_point_at(coords, i, points_num) for i in range(points_num)]


def angles(img: Image):
    image = img.data
    # image = gaussian(image, sigma=0.5)
    con = find_contours(image, .9)
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r)

    contour = con[0]
    coords = approximate_polygon(contour, tolerance=2.75)
    ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=3)
    ax.plot(coords[0, 1], coords[0, 0], '*', color='blue')
    ax.plot(coords[1, 1], coords[1, 0], '*', color='green')
    ang = compute_angles(coords)
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

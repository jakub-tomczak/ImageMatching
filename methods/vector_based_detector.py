from utils.angle_detector import calculate_arms, find_best_bases, get_contours
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import DEBUG
import matplotlib.pyplot as plt
import numpy as np

from utils.model import Angle
from utils.plotting_helper import interpolate_between_points
from utils.points_helpers import get_orthogonal_vector, distance
import cv2


def xy_to_yx(point: [int, int]):
    """
    Swaps x and x values. In some functions openCV returns [x, y] instead of [y, x].
    :param point:
    :return:
    """
    return [point[1], point[0]]


def rotate_image_using_base(image: Image, base: [[int, int]]):
    """
    :param image:
    :param base: [first, second] which are arrays of 2 elements indicating two points of the base. [y, x]
    :return:
    """
    first = base[0]
    second = base[1]
    third = [max(first[1], second[1]), first[1] if first[0] < second[0] else second[1]]

    angle = Angle.calculate_angle_between(first, second, third)
    print("angle is", angle)
    M = cv2.getRotationMatrix2D((image.data.shape[1] / 2, image.data.shape[0] / 2), -angle, 1)
    rotated_image = cv2.warpAffine(image.data, M, image.data.shape)
    cv2.imshow('rotated image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_base_from_box(image: Image, box: [[int, int]]):
    # the highest situated point in the rectangle (we need to compare 2 lengths from 3 points)
    point_to_exclude = np.unravel_index(box[:, 0].argmin(), box[:, 0].shape)
    distances = []

    # find two distances with their indices of the first point in box array
    # assumes that box has 4 rows
    for i in range(len(box)):
        if i <= point_to_exclude[0] <= i+1:
            # don't draw point between the current and the point_to_exclude
            continue
        distances.append([i, distance(box[i], box[(i+1) % len(box)])])

    # sort from the longest to the shortest distance, take the longest one
    longest_distance = sorted(distances, key=(lambda x: x[1]), reverse=True)[0]

    first = box[longest_distance[0]]
    second = box[(longest_distance[0] + 1) % len(box)]

    # plot image with base indicated
    third = [first[0] if first[1] < second[1] else second[0], max(first[1], second[1])]
    fig, ax = plt.subplots()
    ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
    for i in range(len(box)):
        # ax.plot expects x then y value
        ax.plot([box[i][1], box[(i + 1) % len(box)][1]], [box[i][0], box[(i + 1) % len(box)][0]], '-y', linewidth=1)

    ax.plot([first[1], second[1]], [first[0], second[0]], '-r', linewidth=1)
    ax.plot([second[1], third[1]], [second[0], third[0]], '-r', linewidth=1)
    ax.plot([third[1], first[1]], [third[0], first[0]], '-r', linewidth=1)
    plt.show()

    return [first, second]


def plot_box(image: Image, box: [[int, int]]):
    fig, ax = plt.subplots()
    ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
    for i in range(len(box)):
        # ax.plot expects x then y value
        ax.plot([box[i][1], box[(i + 1) % len(box)][1]], [box[i][0], box[(i + 1) % len(box)][0]], '-y', linewidth=1)
    plt.show()


def get_min_area_rectangle_box(image: Image):
    _, contours, hierarchy = cv2.findContours(image.data, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = [xy_to_yx(point) for point in box]
    return np.int0(box)


def get_initial_vertices(image: Image):
    """
    Finds start and end point that constitutes a vector for find deviations.
    :param image:
    :return:
    """
    coords, _ = get_contours(image, True)
    image.points_coords = coords[:-1]
    arms = calculate_arms(image.points_coords)
    base_candidates = find_best_bases(arms)
    image.arms = arms
    if len(base_candidates) > 0:
        most_probable_base = base_candidates[0]
    else:
        if DEBUG:
            print('No base found for image {}.'.format(image.name))
        return None

    start_point = image.arms[(most_probable_base.start - 1) % len(image.arms)].a
    end_point = image.arms[(most_probable_base.end + 1) % len(image.arms)].b
    return start_point, end_point


def calculate_deviation_for_point(image: Image, start: [float, float], orthogonal_vector, length, debug_draw: bool = False):
    # points to check
    yy, xx = interpolate_between_points(start, start + orthogonal_vector, length, target_type=int)
    start_value = image.data[yy[0], xx[0]]

    if debug_draw:
        fig, ax = plt.subplots()
        ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
        ax.plot(xx, yy, '-g', linewidth=1)

    def binary_search(data, start_index, stop_index, initial_value, xx, yy):
        current_index = (start_index+stop_index) // 2
        if debug_draw:
            diff = 1 if current_index+1 < len(xx) else -1
            ax.plot([xx[current_index], xx[current_index+diff]], [yy[current_index], yy[current_index+diff]], 'r', linewidth=3)
        # check whether we are in bounds
        if data.shape[1] < xx[current_index] or xx[current_index] < 0 \
                or data.shape[0] < yy[current_index] or yy[current_index] < 0:
            return None
        if stop_index - start_index <= 1:
            return [current_index, yy[current_index], xx[current_index]]
        if abs(data[yy[current_index], xx[current_index]] - initial_value) < 1e-2:
            # the same
            start_index = current_index
        else:
            stop_index = current_index
        return binary_search(data, start_index, stop_index, initial_value, xx, yy)

    val = binary_search(image.data, 1, length, start_value, xx, yy)
    if debug_draw and val is not None:
        current_index = val[0]
        ax.plot([xx[current_index], xx[current_index + 1]], [yy[current_index], yy[current_index + 1]], 'b', linewidth=3)
        plt.show()
    return val
    # while value_not_found:
    #     if xx[current_index] >= image.data.shape[1] or yy[current_index] >= image.data.shape[0]:
    #         current_index -= 1
    #     if current_index == 0:
    #         return None
    #     # is different value found ?
    #     if abs(image.data[yy[current_index], xx[current_index]] - start_value) > 1e-3:
    #         # divide into int
    #         current_index //= 2
    #         continue
    #     else:
    #         for i in range(current_index, current_index*2):
    #             if image.data[yy[i], xx[i]] != start_value:
    #                 # i-th is the first index with a value different than start_value
    #                 return i-1


def find_deviations_in_cut(image: Image, start_point: [float, float], end_point: [float, float],
                           debug_draw: bool = True) -> [float]:
    number_of_points_in_vector = 20
    orthogonal_vector_length = int(min(image.data.shape) * .5)
    vector = [end_point[0] - start_point[0], end_point[1] - start_point[1]]
    normal_vector_positive = get_orthogonal_vector(vector, length=orthogonal_vector_length)
    yy, xx = interpolate_between_points(start_point, end_point, number_of_points_in_vector)

    # if debug_draw:
    #     fig, ax = plt.subplots()
    #     ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
    #     ax.plot(xx, yy, '-r')
    #
    #     for x, y in zip(xx[1:-1], yy[1:-1]):
    #         ax.plot([x, x + normal_vector_positive[1]], [y, y + normal_vector_positive[0]], '-y', linewidth=1)
    #     plt.show()

    deviations_vector = np.zeros((number_of_points_in_vector, 3))
    for i, point in enumerate(zip(yy[1:-1], xx[1:-1])):
        diff = calculate_deviation_for_point(image, np.array(point), normal_vector_positive, orthogonal_vector_length)
        if diff is None:
            diff = \
                calculate_deviation_for_point(image, np.array(point), -normal_vector_positive, orthogonal_vector_length)
        deviations_vector[i] = diff

    if debug_draw:
        fig, ax = plt.subplots()
        ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
        ax.plot(xx, yy, '-r')

        for x, y, diff in zip(xx[1:-1], yy[1:-1], deviations_vector):
            ax.plot([x, diff[2]], [y, diff[1]], '-y', linewidth=1)
            # ax.plot([x, x - normal_vector_positive[1]], [y, y - normal_vector_positive[0]], '-y', linewidth=1)
        plt.show()
    print('ok')


def find_matching_images(dataset: Dataset):
    for image in dataset.images:
        rect_box = get_min_area_rectangle_box(image)
        # plot_box(image, rect_box)
        find_base_from_box(image, rect_box)
        start, end = get_initial_vertices(image)

        if start is None or end is None:
            print("Couldn't find coords of deviation vector, image {}".format(image.name))
            continue
        find_deviations_in_cut(image, start, end)

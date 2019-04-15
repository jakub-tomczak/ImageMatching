from utils.angle_detector import calculate_arms, find_best_bases, get_contours
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import DEBUG
import matplotlib.pyplot as plt
from utils.plotting_helper import interpolate_between_points
from utils.points_helpers import get_orthogonal_vector


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


def calculate_deviation(image: Image, start: (float, float), normal_vector, length):
    bounded_end_y = min(abs(length), abs(start[0] - image.data.shape[0]))
    bounded_end_x = min(abs(length), abs(start[1] - image.data.shape[1]))


def find_deviations_in_cut(image: Image, start_point: [float, float], end_point: [float, float],
                           debug_draw: bool = True) -> [float]:
    number_of_points_in_vector = 20
    normal_vector_length = min(image.data.shape) * .5
    vector = [end_point[0] - start_point[0], end_point[1] - start_point[1]]
    normal_vector_positive = get_orthogonal_vector(vector, length=normal_vector_length)
    normal_vector_negative = -normal_vector_positive
    xx, yy = interpolate_between_points(start_point, end_point, number_of_points_in_vector)

    if debug_draw:
        fig, ax = plt.subplots()
        ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
        ax.plot(xx, yy, '-r')

        for x, y in zip(xx[1:-1], yy[1:-1]):
            ax.plot([x, x + normal_vector_positive[1]], [y, y + normal_vector_positive[0]], '-y', linewidth=1)
            ax.plot([x, x + normal_vector_negative[1]], [y, y + normal_vector_negative[0]], '-y', linewidth=1)
        plt.show()


def find_matching_images(dataset: Dataset):
    for image in dataset.images:
        start, end = get_initial_vertices(image)

        if start is None or end is None:
            print("Couldn't find coords of deviation vector, image {}".format(image.name))
            continue
        find_deviations_in_cut(image, start, end)

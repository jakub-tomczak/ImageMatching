from utils.angle_detector import calculate_arms, find_best_bases, get_contours
from utils.dataset_helper import Image, Dataset
from utils.debug_conf import DEBUG
import matplotlib.pyplot as plt
from utils.plotting_helper import interpolate_between_points
from utils.points_helpers import get_normal_vector


def get_initial_vertices(image: Image):
    """
    Finds start and end point that constitutes a vector for find deviations.
    :param image:
    :return:
    """
    coords, _ = get_contours(image, False)
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

    # find_base_rotation(most_probable_base.arm)
    start_point = image.arms[(most_probable_base.start - 1) % len(image.arms)].a
    end_point = image.arms[(most_probable_base.end + 1) % len(image.arms)].b

    return start_point, end_point


def find_deviation(start: (float, float), end: (float, float), range):
    pass


def find_deviations_in_cut(image: Image, start_point: [float, float], end_point: [float, float],
                           debug_draw: bool = True) -> [float]:
    number_of_points_in_vector = 20
    normal_vector_length = min(image.data.shape) * .5
    normal_vector_positive = get_normal_vector(start_point, end_point, length=normal_vector_length)
    normal_vector_negative = get_normal_vector(start_point, end_point, length=-normal_vector_length)
    xx, yy = interpolate_between_points(start_point, end_point, number_of_points_in_vector)

    if debug_draw:
        fig, ax = plt.subplots()
        ax.imshow(image.data, interpolation='nearest', cmap=plt.cm.Greys_r)
        ax.plot(xx, yy, '-r')
        for x, y in zip(xx, yy):
            ax.plot([x, x + normal_vector_positive[1]], [y, y + normal_vector_positive[0]], '-b', linewidth=1)
            ax.plot([x, x + normal_vector_negative[1]], [y, y + normal_vector_negative[0]], '-y', linewidth=1)
        plt.show()


def find_matching_images(dataset: Dataset):
    for image in dataset.images:
        start, end = get_initial_vertices(image)
        if start is None or end is None:
            print("Couldn't find coords of deviation vector, image {}".format(image.name))
            continue
        find_deviations_in_cut(image, start, end)

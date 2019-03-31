import argparse
import math

from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.measure import find_contours, approximate_polygon

from utils.dataset_helper import load_dataset, Image, Dataset

debug = True


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
    image = gaussian(image)
    # image = image > 200
    con = find_contours(image,.8)
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r)

    for n, contour in enumerate(con):
        coords = approximate_polygon(contour, tolerance=1)
        ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=3)
        ax.plot(coords[0,1], coords[0,0], '*', color='blue')
        ax.plot(coords[1,1], coords[1,0], '*', color='green')
        ang = compute_angles(coords)
        print(ang)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def parse_dataset(dataset: Dataset):
    for img in dataset.images:
        angles(img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset.")
    parser.add_argument("number_of_images", help="Number of images to be loaded from a dataset.")
    args = parser.parse_args()
    return args.path, int(args.number_of_images)


def main():
    path, number_of_images = parse_args()
    dataset = load_dataset(path, number_of_images)
    parse_dataset(dataset)
    if debug:
        dataset.set_matching_images()
    for image in dataset.images:
        print(f'correct answer for image {image.path} is {image.correct}.')


if __name__ == "__main__":
    main()

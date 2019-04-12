import numpy as np
from matplotlib import pyplot as plt

from utils.dataset_helper import Dataset
from utils.model import Angle, Arm
from utils.plotting_helper import plot_line

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def result(is_ok):
    return (OKGREEN + "OK" if is_ok else FAIL + "FAIL") + ENDC


def calculate_points(ranking, dataset: Dataset):
    correct = [i.correct[0] for i in dataset.images]
    images_num = len(correct)
    points = 0
    for r, c in zip(ranking, correct):
        try:
            actual_index = r.index(c)
        except ValueError:
            actual_index = -1
        if actual_index == -1:
            points += 1 / images_num
        else:
            points += 1 / (actual_index + 1)
    return points


def print_debug_info(dataset: Dataset, ranking):
    dataset.set_matching_images()
    for image, r in zip(dataset.images, ranking):
        cor = image.correct[0]
        ans = r[0]
        print("correct answer for image {} is {}; got {} {}".format(image.name, cor, ans, result(cor == ans)))
    points = calculate_points(ranking, dataset)
    print("received points: {}".format(points))


def show_debug_info(ang: [Angle], arms: [Arm], coords, image, distances, best_candidate_for_base):
    ax = draw_image_spec(image, ang, arms, coords)

    # draw a few longest distances
    for i in range(1):
        color = 'yellow'
        if best_candidate_for_base is not None:
            p_0_index, p_1_index = best_candidate_for_base
        else:
            color = 'blue'
            if i >= len(distances):
                break
            p_0_index = distances[i][1]
            p_1_index = (p_0_index + 1) % len(distances)

        plot_line(ax, coords[p_0_index], coords[p_1_index], color)

    print(ang)
    plt.show()


def draw_image_spec(image, ang: [Angle], arms: [Arm], coords):
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r)
    angles_points = np.array([a.point for a in ang])
    ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=5)
    ax.plot(coords[0, 1], coords[0, 0], '*', color='blue')
    ax.plot(coords[1, 1], coords[1, 0], '*', color='green')
    ax.plot(coords[-2, 1], coords[-2, 0], 'o', color='orange')
    ax.plot(angles_points[:, 1], angles_points[:, 0], 'o', color='green')
    for a in arms:
        c = np.array([a.a, a.b])
        ax.plot(c[:, 1], c[:, 0], '-*', color='pink')
    return ax


def show_comparing_points(img1, img2, points, first_as_first):
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

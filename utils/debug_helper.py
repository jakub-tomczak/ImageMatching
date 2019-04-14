import numpy as np
from matplotlib import pyplot as plt

from utils.dataset_helper import Dataset, RESULT_DISPLAY_CORRECT, RESULT_DISPLAY_POINTS
from utils.model import Angle, Arm, BaseArm
from utils.plotting_helper import plot_line

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'
INFO = '\033[35m'
POINTS_INFO = '\033[1;36m'


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
        if RESULT_DISPLAY_CORRECT:
            print("correct answer for image {} is {}; got {} {}".format(image.name, cor, ans, result(cor == ans)))
    if RESULT_DISPLAY_POINTS:
        points = calculate_points(ranking, dataset)
        total_points = len(dataset.images)
        print("received points: {}{}/{} ({}%){}".format(
            POINTS_INFO, points, total_points, points / total_points * 100, ENDC)
        )


def show_debug_info(ang: [Angle], arms: [Arm], coords, image, best_bases: [BaseArm]):
    ax = draw_image_spec(image, ang, arms, coords)

    for a in best_bases:
        color = 'yellow'
        plot_line(ax, a.arm.a, a.arm.b, color)

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


def show_comparing_points(img1, f_angles, img2, s_angles, points, first_as_first):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(img1, interpolation='nearest', cmap=plt.cm.Greys_r)
    ax2.imshow(img2, interpolation='nearest', cmap=plt.cm.Greys_r)
    angles_points = np.array([a.point for a in f_angles])
    ax1.plot(angles_points[:, 1]/4, angles_points[:, 0]/4, 'o', color='yellow')
    angles_points = np.array([a.point for a in s_angles])
    ax2.plot(angles_points[:, 1]/4, angles_points[:, 0]/4, 'o', color='yellow')
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

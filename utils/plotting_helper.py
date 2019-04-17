import numpy as np


def plot_line(ax, coord_1, coord_2, line_color):
    xx = [coord_1[1], coord_2[1]]
    yy = [coord_1[0], coord_2[0]]
    ax.plot(xx, yy, 'ro-', color=line_color)


def interpolate_between_points(coord1: [float, float], coord2: [float, float], number_of_points: int,
                               target_type=float) -> ([float], [float]):
    xx = np.linspace(coord1[1], coord2[1], number_of_points, dtype=target_type)
    yy = np.linspace(coord1[0], coord2[0], number_of_points, dtype=target_type)
    return yy, xx

import matplotlib.pyplot as plt

def plot_line(ax, coord_1, coord_2, line_color):
    xx = [coord_1[1], coord_2[1]]
    yy = [coord_1[0], coord_2[0]]
    ax.plot(xx, yy, 'ro-', color=line_color)
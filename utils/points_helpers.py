def accumulate_points(points):
    x_sum = 0
    y_sum = 0
    total = len(points)
    for [y, x] in points:
        x_sum += x
        y_sum += y
    return [y_sum / total, x_sum / total]


def distance(a, b):
    return pitagoras(a[1] - b[1], a[0] - b[0])


def pitagoras(a, b):
    return pow(a ** 2 + b ** 2, 0.5)

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


def take_two_subseqent_points_indices(point_index: int, number_of_points: int, is_other_next_point: bool) -> (int, int):
    """
    takes returns tuple (point, other_point)
    other point is the next one if is_other_next_point
    other point is the previous one if not is_other_next_point
    :param point_index:
    :param number_of_points:
    :param is_other_next_point:
    :return:
    """
    if is_other_next_point:
        return point_index, (point_index + 1) % (number_of_points - 1)
    else:
        return (point_index - 1) % (number_of_points - 1), point_index


def get_normal_vector(start_point, end_point, length: float = 1.0):
    """
    :param start_point: Tuple or list (y, x)
    :param end_point: Tuple or list (y, x)
    :param length: Optional length of the returning vector.
    :return: Normal vector (length 1) as Tuple (y, x)
    """
    x_0 = abs(start_point[1] - end_point[1])
    y_0 = abs(start_point[0] - end_point[0])

    coeff = y_0 / x_0
    y = pow((1.0 / (coeff ** 2 + 1)), .5) * length
    x = -coeff * y

    return y, x

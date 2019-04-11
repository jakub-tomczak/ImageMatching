from utils.points_helpers import accumulate_points, distance


def compress_points(coords, min_distance):
    buffer = []
    points = []

    def last_center_item():
        return buffer[int(len(buffer) - 1 / 2)]

    for c in coords:
        if len(buffer) > 0:
            dist = distance(c, last_center_item())
            if dist > min_distance:
                points.append(accumulate_points(buffer))
                buffer = []
        buffer.append(c)
    if len(points) > 0:
        first = points[0]
        dist = distance(first, last_center_item())
        if dist <= min_distance:
            points.pop(0)
            buffer.append(first)
    points.append(accumulate_points(buffer))
    return points

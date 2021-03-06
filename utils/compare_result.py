from utils.model import ComparePoint, ImageAngleData


class CompareResult:
    def __init__(self, first: ImageAngleData, second: ImageAngleData):
        different_offsets = []
        for f_angles in first.comparison_points:
            for s_angles in second.comparison_points:
                shorter, longer, first_as_first = (f_angles, s_angles, True) \
                    if len(f_angles) < len(s_angles) \
                    else (s_angles, f_angles, False)
                shorter_len = len(shorter)
                longer_len = len(longer)
                a_i = 0
                b_i = longer_len - 1
                total_sim = 0
                while a_i < shorter_len and b_i >= 0:
                    ap, bp, sim = CompareResult.find_matching_angle(shorter, longer, a_i, b_i)
                    a_i += ap + 1
                    b_i -= bp + 1
                    total_sim += sim

                different_offsets.append(total_sim / max(longer_len, 1))
        self.similarity = max(different_offsets)

    @staticmethod
    def find_matching_angle(a: [ComparePoint], b: [ComparePoint], a_i: int, b_i: int):
        a_point, b_point, a_off, b_off = CompareResult.get_aligned_points(a, b, a_i, b_i)
        if a_point is None:
            return a_off, b_off, 0

        if a_point.can_compare_with(b_point):
            a0, b0, sim = a_point.similarity(b_point)
            return a0 + a_off, b0 + b_off, sim
        else:
            if b_i - b_off > 0:
                b_point2 = b[b_i - 1 - b_off]
                if a_point.can_compare_with(b_point2):
                    a0, b0, sim = a_point.similarity(b_point2)
                    return a0 + a_off, b0 + 1 + b_off, sim
            if a_i + a_off < len(a) - 1:
                a_point2 = a[a_i + 1 + a_off]
                if a_point2.can_compare_with(b_point):
                    a0, b0, sim = a_point2.similarity(b_point)
                    return a0 + 1 + a_off, b0 + b_off, sim
        return a_off, b_off, 0

    @staticmethod
    def get_aligned_points(a: [ComparePoint], b: [ComparePoint], a_i, b_i):
        a_point = a[a_i]
        b_point = b[b_i]
        a_off = 0
        b_off = 0
        range_com = 0.075
        len_a = len(a)
        dif = a_point.progress_difference(b_point)

        while dif > range_com or dif < -range_com:
            if dif < -range_com:
                if a_i + a_off + 1 >= len_a:
                    return None, None, a_off, b_off
                a_off += 1
                a_point = a[a_i + a_off]
            elif dif > range_com:
                if b_i - b_off - 1 < 0:
                    return None, None, a_off, b_off
                b_off += 1
                b_point = b[b_i - b_off]
            dif = a_point.progress_difference(b_point)
        return a_point, b_point, a_off, b_off

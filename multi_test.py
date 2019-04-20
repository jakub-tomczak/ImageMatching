import os
from time import time
from main import run, RESULT_DISPLAY_TOTAL_POINTS, method_one, method_two
from utils.debug_helper import INFO, ENDC, get_collected_points, TOTAL_INFO


def test_instance(dirname, images):
    start = time()
    run(dirname, images, True, False, method=method_one)
    time_passed = time() - start
    print('{0}set: {1}\t\ttime: {2:0.3f}{3}'.format(INFO, dirname, time_passed, ENDC))
    return time_passed


if __name__ == '__main__':
    to_test = []
    for dirname, dirnames, filenames in os.walk('./data'):
        images = len(list(filter(lambda name: name.endswith(".png"), filenames)))
        if images == 0:
            continue
        to_test.append((dirname, images))
    to_test = sorted(to_test, key=lambda o: o[0])

    times = [test_instance(dirname, images) for dirname, images in to_test]

    if RESULT_DISPLAY_TOTAL_POINTS:
        points, total = get_collected_points()
        print("TOTAL: {}{}/{} ({}%){}".format(
            TOTAL_INFO, round(points, 4), total, round(points / total * 100, 4), ENDC)
        )
        print("TOTAL TIME: {0}{1:0.3f}{2}".format(
            TOTAL_INFO, sum(times), ENDC)
        )
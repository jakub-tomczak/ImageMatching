import os

from main import run, RESULT_DISPLAY_TOTAL_POINTS
from utils.debug_helper import INFO, ENDC, get_collected_points, TOTAL_INFO

if __name__ == '__main__':
    to_test = []
    for dirname, dirnames, filenames in os.walk('./data'):
        images = len(list(filter(lambda name: name.endswith(".png"), filenames)))
        if images == 0:
            continue
        to_test.append((dirname, images))
    to_test = sorted(to_test, key=lambda o: o[0])
    for dirname, images in to_test:
        print(INFO + dirname + ENDC)
        run(dirname, images, True, False)

    if RESULT_DISPLAY_TOTAL_POINTS:
        points, total = get_collected_points()
        print("TOTAL: {}{}/{} ({}%){}".format(
            TOTAL_INFO, round(points, 4), total, round(points / total * 100, 4), ENDC)
        )

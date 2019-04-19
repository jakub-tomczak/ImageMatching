import argparse

from utils.angle_detector import get_ranking
from utils.dataset_helper import load_dataset
from utils.debug_conf import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset.")
    parser.add_argument("number_of_images", help="Number of images to be loaded from a dataset.")
    args = parser.parse_args()
    return args.path, int(args.number_of_images)


def run(path, number_of_images, debug=False, display_ranking=True):
    dataset = load_dataset(path, number_of_images)
    ranking = get_ranking(dataset)
    if display_ranking:
        for r in ranking:
            print(" ".join(str(i) for i in r))
    if debug:
        from utils.debug_helper import print_debug_info
        print_debug_info(dataset, ranking)


if __name__ == "__main__":
    path, number_of_images = parse_args()
    run(path, number_of_images, DEBUG, RESULT_DISPLAY_RANKING)

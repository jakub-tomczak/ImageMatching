import argparse

from methods.vector_based_detector import find_matching_images
from utils.angle_detector import get_ranking
from utils.dataset_helper import Dataset
from utils.dataset_helper import load_dataset
from utils.debug_conf import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset.")
    parser.add_argument("number_of_images", help="Number of images to be loaded from a dataset.")
    args = parser.parse_args()
    return args.path, int(args.number_of_images)


def process_ranking(dataset: Dataset, ranking: [[int]], debug: bool, display_ranking: bool):
    if display_ranking:
        for r in ranking:
            print(" ".join(str(i) for i in r))
    if debug:
        from utils.debug_helper import print_debug_info
        print_debug_info(dataset, ranking)


def method_one(dataset: Dataset):
    return get_ranking(dataset)


def method_two(dataset: Dataset):
    return find_matching_images(dataset)


def run(path, number_of_images, debug=False, display_ranking=True, method=None):
    dataset = load_dataset(path, number_of_images)

    if method is None:
        method = method_two
    ranking = method(dataset)
    process_ranking(dataset, ranking, debug, display_ranking)


if __name__ == "__main__":
    path, number_of_images = parse_args()
    run(path, number_of_images, DEBUG, RESULT_DISPLAY_RANKING)

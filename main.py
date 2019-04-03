import argparse

from utils.angle_detector import get_ranking
from utils.dataset_helper import load_dataset
from utils.debug_helper import print_debug_info

debug = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset.")
    parser.add_argument("number_of_images", help="Number of images to be loaded from a dataset.")
    args = parser.parse_args()
    return args.path, int(args.number_of_images)


def main():
    path, number_of_images = parse_args()
    dataset = load_dataset(path, number_of_images)
    ranking = get_ranking(dataset)
    for r in ranking:
        print(" ".join(str(i) for i in r))
    if debug:
        print_debug_info(dataset, ranking)


if __name__ == "__main__":
    main()

from matplotlib import pyplot as plt
from utils.dataset_helper import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset.")
    parser.add_argument("number_of_images", help="Number of images to be loaded from a dataset.")
    args = parser.parse_args()
    return (args.path, int(args.number_of_images))

def main():
    path, number_of_images = parse_args()
    dataset = load_dataset(path, number_of_images)

if __name__ == "__main__":
    main()
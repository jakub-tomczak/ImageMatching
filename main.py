import argparse

from utils.angle_detector import get_ranking
from utils.dataset_helper import load_dataset, Dataset

debug = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset.")
    parser.add_argument("number_of_images", help="Number of images to be loaded from a dataset.")
    args = parser.parse_args()
    return args.path, int(args.number_of_images)


def calculate_points(ranking, dataset: Dataset):
    correct = [i.correct[0] for i in dataset.images]
    images_num = len(correct)
    points = 0
    for r, c in zip(ranking, correct):
        actual_index = r.index(c)
        if actual_index == -1:
            points += 1 / images_num
        else:
            points += 1 / (actual_index + 1)
    return points


def main():
    path, number_of_images = parse_args()
    dataset = load_dataset(path, number_of_images)
    ranking = get_ranking(dataset)
    for r in ranking:
        print(" ".join(str(i) for i in r))
    if debug:
        dataset.set_matching_images()
        for image in dataset.images:
            print(f'correct answer for image {image.name} is {image.correct}.')
        points = calculate_points(ranking, dataset)
        print(f'received points: {points}')


if __name__ == "__main__":
    main()

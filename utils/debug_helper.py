from utils.dataset_helper import Dataset

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def result(is_ok):
    return (OKGREEN + "OK" if is_ok else FAIL + "FAIL") + ENDC


def calculate_points(ranking, dataset: Dataset):
    correct = [i.correct[0] for i in dataset.images]
    images_num = len(correct)
    points = 0
    for r, c in zip(ranking, correct):
        try:
            actual_index = r.index(c)
        except ValueError:
            actual_index = -1
        if actual_index == -1:
            points += 1 / images_num
        else:
            points += 1 / (actual_index + 1)
    return points


def print_debug_info(dataset: Dataset, ranking):
    dataset.set_matching_images()
    for image, r in zip(dataset.images, ranking):
        cor = image.correct[0]
        ans = r[0]
        print("correct answer for image {} is {}; got {} {}".format(image.name, cor, ans, result(cor == ans)))
    points = calculate_points(ranking, dataset)
    print("received points: {}".format(points))

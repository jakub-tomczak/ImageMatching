import skimage.io as io
from os.path import join, isfile
from utils.debug_conf import *

dataset_directory = 'data'
extension = 'png'
correct_filename = 'correct.txt'


class Image:
    def __init__(self, path, data, name):
        self.path = path
        self.data = data
        self.name = name
        # assumes that there may be more than one correct answer
        # which is less likely
        self.correct = []
        self.points_coords = []
        self.base_coords = ()
        self.arms = []

    def set_matching_images(self, matching):
        self.correct = [int(x) for x in matching.split(' ')]


class Dataset:
    def __init__(self, directory, number_of_images):
        print('Loading dataset from: {}, images to load: {}'.format(directory, number_of_images))
        self.directory = directory
        self.number_of_images = number_of_images
        self.images_to_load = [join(directory, '{}.{}'.format(x, extension)) for x in range(number_of_images)]
        self.images = [Image(image, load_image(image), i) for i, image in enumerate(self.images_to_load)]

    def set_matching_images(self):
        match_image_file_path = join(self.directory, correct_filename)
        if isfile(match_image_file_path):
            with open(match_image_file_path, 'r') as f:
                [self.images[i].set_matching_images(f.readline()) for i in range(self.number_of_images)]


def load_image(dataset, image_name):
    return load_image(join(dataset_directory, dataset, image_name))


def load_image(path):
    if DEBUG:
        print('loading image: {}'.format(path))
    return io.imread(path, as_gray=True)


def load_dataset(directory, number_of_images):
    return Dataset(directory, number_of_images)

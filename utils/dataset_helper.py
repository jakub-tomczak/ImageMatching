
import skimage.io as io
from os.path import join

dataset_directory = 'data'
extension = 'png'

class Dataset:
    def __init__(self, directory, number_of_images):
        print('Loading dataset from: {}, images to load: {}'.format(directory, number_of_images))
        self.directory = directory
        self.number_of_images = number_of_images
        self.images_to_load = [join(directory, '{}.{}'.format(x, extension)) for x in range(number_of_images)]
        self.images = [load_image(image) for image in self.images_to_load]

def load_image(dataset, image_name):
    return load_image(join(dataset_directory, dataset, image_name))

def load_image(path):
    print('loading image: {}'.format(path))
    return io.imread(path, as_gray=True)

def load_dataset(directory, number_of_images):
    return Dataset(directory, number_of_images)

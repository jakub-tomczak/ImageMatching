import os

from main import run
from utils.debug_helper import INFO, ENDC

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

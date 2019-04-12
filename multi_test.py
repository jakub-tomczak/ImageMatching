import os

from main import run
from utils.debug_helper import INFO, ENDC

if __name__ == '__main__':
    for dirname, dirnames, filenames in os.walk('./data'):
        images = len(list(filter(lambda name: name.endswith(".png"), filenames)))
        if images == 0:
            continue
        print("TESTING " + INFO + dirname + ENDC)
        run(dirname, images, True)

#!/bin/bash
IMG_NAME=python
IMG_TAG=piro
if [[ "$(sudo docker images -q $IMG_NAME:$IMG_TAG 2> /dev/null)" == "" ]]; then
  echo "Building $IMG_NAME:$IMG_TAG..."
  sudo docker build -t $IMG_NAME:$IMG_TAG .
fi
sudo docker run -it -v $(pwd):/piro --rm $IMG_NAME:$IMG_TAG

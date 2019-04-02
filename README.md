Prerequisites:
```
pip install -r requirements.txt --user
sudo apt -y install python3-tk
```

How to run:
```
python3 main.py path/to/dataset number_of_images
```

## Docker

### Pull from docker hub
Docker image is available on docker hub, to pull it from there use:
```
docker pull jakubtomczak/python-computer-vision
```

### Build from Dockerfile
Building Dockerfile:
```
sudo docker build -t python:piro .
```

Run docker image:
```
sudo docker run -it --rm -v $(pwd):/piro python:piro /bin/bash
```
and then
run `image matching` inside docker image:
```
cd /piro
python main.py path/to/dataset number_of_images
```

### Build and run from script
Use `run_docker.sh` to build and run image.
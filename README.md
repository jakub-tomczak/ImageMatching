Prerequisites:
```
pip install -r requirements.txt --user
sudo apt -y install python3-tk
```

How to run:
```
python3 main.py path/to/dataset number_of_images
```

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
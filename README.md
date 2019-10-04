# Jetson Object Detection

# To build and run the CPU accelerated container
sudo docker build . -f ./docker/Dockerfile.cpu -t object-detection-cpu
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 object-detection-cpu

# To build and run the GPU accelerated container
sudo docker build . -f ./docker/Dockerfile.gpu -t object-detection-gpu
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 object-detection-gpu

> Run from the root folder of the repo
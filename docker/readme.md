## Start with building the base image
sudo docker build . -f ./docker/Dockerfile.base.dev -t base-dev

# To build and run the opencv container
sudo docker build . -f ./docker/Dockerfile.opencv-detector -t opencv-detector
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 opencv-detector

# To build and run the tensorflow container
sudo docker build . -f ./docker/Dockerfile.tensorflow-detector-t tensorflow-detector
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 tensorflow-detector

# To build and run the tensorrt detector
sudo docker build . -f ./docker/Dockerfile.tensorrt-detector-t tensorrt-detector
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 tensorrt-detector

> Run from the root folder of the repo
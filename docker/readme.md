# To build and run the opencv container
sudo docker build . -f ./docker/Dockerfile.opencv -t opencv
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 opencv

# To build and run the tensorflow container
sudo docker build . -f ./docker/Dockerfile.tensorflow -t tensorflow
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 tensorflow

# To build and run the trt converter container
sudo docker build . -f ./docker/Dockerfile.trtconverter -t trtconverter
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 trtconverter

> Run from the root folder of the repo
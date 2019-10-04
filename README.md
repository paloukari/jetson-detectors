# High Throuput Object detector for the Nvidia Jetsoin devices family 

This is the code used in [this blog post](http://havedatawilltrain.com/three-threads-to-perdido)

## Clone the example repo
```
https://github.com/paloukari/jetson-detectors
cd jetson-detectors
```

## To build and run the CPU accelerated container
```
sudo docker build . -f ./docker/Dockerfile.cpu -t object-detection-cpu
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 object-detection-cpu
```


## To build and run the GPU accelerated container
```
sudo docker build . -f ./docker/Dockerfile.gpu -t object-detection-gpu
sudo docker run --rm --runtime nvidia --privileged -ti -e DISPLAY=$DISPLAY -v "$PWD":/src -p 32001:22 object-detection-gpu
```

> Run from the root folder of the repo

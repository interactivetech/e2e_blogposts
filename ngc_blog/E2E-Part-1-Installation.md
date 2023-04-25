# Part 1 End-to-End Example training object detection model using NVIDIA Pytorch Container from NGC
## Installation
 ----

Note this Object Detection demo is based on https://github.com/pytorch/vision/tree/v0.11.3

This notebook walks you each step to train a model using containers from the NGC Catalog. We chose the GPU optimized Pytorch container as an example. The basics of working with docker containers apply to all NGC containers.

We will show you how to:

* Install the Docker Engine on your system
* Pull a Pytorch container from the NGC Catalog using Docker
* Run the Pytorch container using Docker

Let's get started!

---

### 1. Install the Docker Engine
Go to https://docs.docker.com/engine/install/ to install the Docker Engine on your system.


### 2. Download the TensorFlow container from the NGC Catalog 

Once the Docker Engine is installed on your machine, visit https://ngc.nvidia.com/catalog/containers and search for the TensorFlow container. Click on the TensorFlow card and copy the pull command.
UPDATE IMG
<img src="https://raw.githubusercontent.com/kbojo/images/master/NGC.png">

Open the command line of your machine and past the pull command into your command line. Execute the command to download the container. 

```$ docker pull nvcr.io/nvidia/pytorch:21.11-py3```
    
The container starts downloading to your computer. A container image consists of many layers; all of them need to be pulled. 

### 3. Run the TensorFlow container image

Once the container download is completed, run the following code in your command line to run and start the container:

```$ docker run -it --gpus all  -p 8888:8888 -v $PWD:/projects --network=host nvcr.io/nvidia/pytorch:21.11-py3```
UPDATE IMG
<img src="https://raw.githubusercontent.com/kbojo/images/master/commandline1.png">

### 4. Install Jupyter lab and open a notebook

Within the container, run the following commands:

```pip install torchvision==0.11.3 jupyterlab```

```jupyter lab --ip=0.0.0.0 --port=8888 --allow-root```

Open up your favorite browser and enter: http://localhost:8888/?token=*yourtoken*.
UPDATE IMG
<img src="https://raw.githubusercontent.com/kbojo/images/master/commandline2.png">

You should see the Jupyter Lab application. Click on the plus icon to launch a new Python 3 noteb
ook.

Follow along with the image classification with the TensorFlow example provided below.


```python

```

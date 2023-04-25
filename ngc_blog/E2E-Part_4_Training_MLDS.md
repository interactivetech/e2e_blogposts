# E2E Part 4 - Training on Machine Learning Development System (MLDS)
 ----
This notebook walks you the commands to run the same training as Step 3, but using the Machine Learing Development (using the PyTorchTrial API). 
All the code is configured to run out of the box. The main change is defining a `class ObjectDetectionTrial(PyTorchTrial)` to incorporate the model, optimizer, dataset, and other training loop essentials.
You can view implementation details looking at `determined_files/model_def.py`

We will show you how to:

* Run a distributed training experiemnt
* Run a distributed hyperparameter search

Note: This notebook was tested on a deployed Machine Learning Development System (MLDS) cluster, running the Machine Learning Development Environment (MLDE) (0.21.2-dev0).

Let's get started!

 ----

# Pre-req: Run startup-hook.sh

This script will install some python dependencies, and install dataset labels needed when loading the Xview dataset:
```bash
# Temporary disable for Grenoble Demo
wget "https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/train_sliced_no_neg/train_300_02.json"
mkdir /tmp/train_sliced_no_neg/
mv train_300_02.json /tmp/train_sliced_no_neg/train_300_02.json 
wget "https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/val_sliced_no_neg/val_300_02.json"
mkdir /tmp/val_sliced_no_neg
mv val_300_02.json /tmp/val_sliced_no_neg/val_300_02.json
```

Note that completing this tutorial requires you to upload your dataset from Step 2 into a publically accessible S3 bucket. This will enable for a large scale distributed experiment to have access to the dataset without installing the dataset on device. View these links to learn how to upload your dataset to an S3 bucket. Review the `S3Backend` class in `data.py`
* https://docs.determined.ai/latest/training/load-model-data.html#streaming-from-object-storage
* https://codingsight.com/upload-files-to-aws-s3-with-the-aws-cli/

When you define your S3 bucket and uploaded your dataset, make sure to change the `TARIN_DATA_DIR` in `build_training_data_loader` with the defined path in the S3 bucket.

```python
def build_training_data_loader(self) -> DataLoader:
    # CHANGE TRAIN_DATA_DIR with different path on S3 bucket
    TRAIN_DATA_DIR='determined-ai-xview-coco-dataset/train_sliced_no_neg/train_images_300_02/'

    dataset, num_classes = build_xview_dataset(image_set='train',args=AttrDict({
                                            'data_dir':TRAIN_DATA_DIR,
                                            'backend':'aws',
                                            'masks': None,
                                            }))
    print("--num_classes: ",num_classes)

    train_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = DataLoader(
                             dataset, 
                             batch_sampler=None,
                             shuffle=True,
                             num_workers=self.hparams.num_workers, 
                             collate_fn=unwrap_collate_fn)
    print("NUMBER OF BATCHES IN COCO: ",len(data_loader))# 59143, 7392 for mini coco
```


```python
!bash demos/xview-torchvision-coco/startup-hook.sh 
```

# Define environment variable DET_MASTER and login in terminal
Run the below commands in a terminal, and complete logging into the determined cluster by chaning <username> to your username.
* `export DET_MASTER=10.182.1.43`
* `det user login <username>`

# Define Determined Experiment

In Determined, a trial is a training task that consists of a dataset, a deep learning model, and values for all of the model’s hyperparameters. An experiment is a collection of one or more trials: an experiment can either train a single model (with a single trial), or can train multiple models via. a hyperparameter sweep a user-defined hyperparameter space.

Here is what a configuration file looks like for a distributed training experiment.

Below is what the `determined_files/const-distributed.yaml` contents look like. `slots_per_trial: 8` defines that we will use 8 GPUs for this experiment.

Edit the change the workspace and project settings in the `determined_files/const-distributed.yaml` file

```yaml
name: resnet_fpn_frcnn_xview_dist_warmup
workspace: <WORKSPACE_NAME>
project: <PROJECT_NAME>
profiling:
 enabled: true
 begin_on_batch: 0
 end_after_batch: null
hyperparameters:
    lr: 0.01
    momentum: 0.9
    global_batch_size: 128
    # global_batch_size: 16
    weight_decay: 1.0e-4
    gamma: 0.1
    warmup: linear
    warmup_iters: 200
    warmup_ratio: 0.001

    step1: 18032 # 14 epochs: 14*1288 == 18,032
    step2: 19320 # 15 epochs: 15*1288 == 19,320
    model: fasterrcnn_resnet50_fpn
    # Dataset
    dataset_file: coco
    backend: aws # specifiy the backend you want to use.  one of: gcs, aws, fake, local
    data_dir: determined-ai-coco-dataset # bucket name if using gcs or aws, otherwise directory to dataset
    masks: false
    num_workers: 4
    device: cuda
environment:
    image: determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-mpi-0.19.10
    environment_variables:                                                                          
        - NCCL_DEBUG=INFO                                                                           
        # You may need to modify this to match your network configuration.                          
        - NCCL_SOCKET_IFNAME=ens,eth,ib
bind_mounts:
    - host_path: /tmp
      container_path: /data
      read_only: false
scheduling_unit: 400
min_validation_period:
    batches: 1288 # For training

searcher:
  name: single
  metric: mAP
  smaller_is_better: true
  max_length:
    batches: 38640 # 30*1288 == 6440# Real Training
records_per_epoch: 1288
resources:
    slots_per_trial: 8
    shm_size: 2000000000
max_restarts: 0

entrypoint: python3 -m determined.launch.torch_distributed --trial model_def:ObjectDetectionTrial

```


```python
# Run the below cell to kick off an experiment
!det e create determined_files/const-distributed.yaml determined_files/
```

    Preparing files to send to master... 237.5KB and 36 files
    Created experiment 77


# Launching a distributed Hyperparameter Search Experiment
To implement an automatic hyperparameter tuning experiment, we need to define the hyperparameter space, e.g., by listing the decisions that may impact model performance. We can specify a range of possible values in the experiment configuration for each hyperparameter in the search space.

View the `x.yaml` file that defines a hyperparameter search where we find the model architecture that achieves the best performance on the dataset.

```yaml
name: xview_frxnn_search
workspace: Andrew
project: Xview FasterRCNN
profiling:
 enabled: true
 begin_on_batch: 0
 end_after_batch: null
hyperparameters:
    lr: 0.01
    momentum: 0.9
    global_batch_size: 128
    weight_decay: 1.0e-4
    gamma: 0.1
    warmup: linear
    warmup_iters: 200
    warmup_ratio: 0.001
    step1: 18032 # 14 epochs: 14*1288 == 18,032
    step2: 19320 # 15 epochs: 15*1288 == 19,320
    model:
      type: categorical
      vals: ['fasterrcnn_resnet50_fpn','fcos_resnet50_fpn', 'ssd300_vgg16','ssdlite320_mobilenet_v3_large','resnet152_fasterrcnn_model','efficientnet_b4_fasterrcnn_model','convnext_large_fasterrcnn_model','convnext_small_fasterrcnn_model']

    # Dataset
    dataset_file: coco
    backend: aws # specifiy the backend you want to use.  one of: gcs, aws, fake, local
    data_dir: determined-ai-coco-dataset # bucket name if using gcs or aws, otherwise directory to dataset
    masks: false
    num_workers: 4

    device: cuda
environment:
    environment_variables:                                                                          
        - NCCL_DEBUG=INFO                                                                           
        # You may need to modify this to match your network configuration.                          
        - NCCL_SOCKET_IFNAME=ens,eth,ib
bind_mounts:
    - host_path: /tmp
      container_path: /data
      read_only: false
scheduling_unit: 400
# scheduling_unit: 40
min_validation_period:
    batches: 1288 # For Real training
searcher:
  name: grid
  metric: mAP
  smaller_is_better: false
  max_length:
    batches: 51520 # 50*1288 == 51520# Real Training

records_per_epoch: 1288
resources:
    # slots_per_trial: 16
    slots_per_trial: 8
    shm_size: 2000000000
max_restarts: 0

entrypoint: python3 -m determined.launch.torch_distributed --trial model_def:ObjectDetectionTrial

```


```python
# Run the below cell to run a hyperparameter search experiment
!det e create determined_files/const-distributed-search.yaml determined_files/
```

    Preparing files to send to master... 312.2KB and 40 files
    Created experiment 79


# Load Checkpoint of Trained Experiment.
Replace the `<EXP_ID>` and run the below cells with the experiment ID once the experiment is completed.


```python
from determined_files.utils.model import build_frcnn_model
from utils import load_model_from_checkpoint
from determined.experimental import Determined,client
```


```python
experiment_id = 76
MODEL_NAME = "xview-fasterrcnn"
checkpoint = client.get_experiment(experiment_id).top_checkpoint(sort_by="mAP", smaller_is_better=False)
print(checkpoint.uuid)
loaded_model = load_model_from_checkpoint(checkpoint)
```

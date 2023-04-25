# Part 2: Data Preparation
 ----

Note this Demo is based on ngc docker image `nvcr.io/nvidia/pytorch:21.11-py3`

This notebook walks you each step to train a model using containers from the NGC Catalog. We chose the GPU optimized Pytorch container as an example. The basics of working with docker containers apply to all NGC containers.

We will show you how to:

* Download the Xview Dataset
* How to convert labels to coco format
* How to conduct the preprocessing step ,Tiling: slicing large satellite imagery into chunks 
* How to upload to s3 bucket to support distributed training

Let's get started!

---


### 2. Pre-reqs, set up jupyter notebook environment using NGC container 

# Execute docker run to create NGC environment for Data Prep
make sure to map host directory to docker directory, we will use the host directory again to 
* `docker run   --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu:/home/ubuntu  -p 8008:8888 -it nvcr.io/nvidia/pytorch:21.11-py3  /bin/bash`

# Run jupyter notebook command within docker container to access it on your local browser
* `cd /home/ubuntu`
* `jupyter lab --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''` 
* `git clone https://github.com/interactivetech/e2e_blogposts.git`



### 0. Download the Xview Dataset
The dataset we will be using is from the DIUx xView 2018 Challenge https://challenge.xviewdataset.org by U.S. National Geospatial-Intelligence Agency (NGA). You will need to create an account at https://challenge.xviewdataset.org/welcome, agree to the terms and conditions, and download the dataset manually.

You can download the dataset at the url https://challenge.xviewdataset.org/data-download




```python
# run pip install to get the SAHI library
!pip install sahi scikit-image opencv-python-headless==4.5.5.64
```


```python
# Example command to download train images with wget command, you will need to update the url as the token is expired"
!wget -O train_images.tgz "https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1680923794&Signature=pn0R9k3BpSukGEdjcNx7Kvs363HWkngK8sQLHxkDOqqkDAHSOCDBmAMAsBhYZ820uMpyu4Ynp1UAV60OmUURyvGorfIRaVF~jJO8-oqRVLeO1f24OGCQg7HratHNUsaf6owCb8XXy~3zaW15FcuORuPV-2Hr6Jxekwcdw9D~g4M2dLufA~qBfTLh3uNjWK5UCAMvyPz2SRLtvc3JLzGYq1eXiKh1dI9W0DyWXov3mVDpBdwS84Q21S2lVi24KJsiZOSJqozuvahydW2AuR~tbXTRbYtmAyPF9ZqT8ZCd9MLeKw2qQJjb7tvzaSZ0F9zPjm2RS8961bo6QoBVeo6kzA__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ"
```


```python
# Example command to download train images with wget command, you will need to update the url as the token is expired"
!wget -O train_labels.tgz "https://d307kc0mrhucc3.cloudfront.net/train_labels.tgz?Expires=1680923794&Signature=YEX~4gioZ7J0pAjEPx7BjJfnOa2j412mx2HlStlqa0cHj-T0T21vo17S8Fs71DXgPlZ5qnIre2-icc7wQ~EuQV-HL1ViS8qH1Aubgj9i0pnHZL07ktiyulX7QStOLywxJ7bOOmQ37iFF~-OcJW3MZfQCTWrP~LdlZMmXz0yGs5WEIYeMyvfUfIhGvrpHcJ14Z3czasSMeOKfwdQsUJoRcFTbmlbZk98IVeEWjmnGTfxGbPBdMmQ96XdT4NohggtzGdqeZhGNfwm7dKGSUbXvGCoFe~fIjBz0~5BvB6rNIaMaFuBA6aGTbCLeG8FlvijcECouhZdMTHmQUlgtSlZjGw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ"
```


```python
# unzip images and labels from /home/ubuntu/e2e_blogposts/ngc_blog
!tar -xf train_images.tgz -C xview_dataset/
```


```python
# unzip labels from /home/ubuntu/e2e_blogposts/ngc_blog directory 
!tar -xf train_labels.tgz -C xview_dataset/
```

# 1. Convert TIF to RGB


```python
# Here loop through all the images and convert them to RGB, this is important for tiling the images and training with pytorch
# will take about an hour to complete
!python data_utils/tif_2_rgb.py --input_dir xview_dataset/train_images \
  --out_dir xview_dataset/train_images_rgb/
```

# 2. How to convert labels to coco format
Here we run a script to convert the dataset labels from .geojson format to COCO format. More details on the COCO format here: 

The result will be two files (in COCO formal) generated `train.json` and `val.json`


```python
# make sure train_images_dir is pointing to the .tif images
!python data_utils/convert_geojson_to_coco.py --train_images_dir xview_dataset/train_images/ \
  --train_images_dir_rgb xview_dataset/train_images_rgb/ \
  --train_geojson_path xview_dataset/xView_train.geojson \
  --output_dir xview_dataset/ \
  --train_split_rate 0.75 \
  --category_id_remapping data_utils/category_id_mapping.json \
  --xview_class_labels data_utils/xview_class_labels.txt

```

# 3. Slicing/Tiling the Dataset
Here we are using the SAHI library to slice our large satellite images. Satellite images can be up to 50k^2 pixels in size, which wouldnt fit in GPU memory. We alleviate this problem by slicing the image. 


```python
!python data_utils/slice_coco.py --image_dir xview_dataset/train_images_rgb/ \
  --train_dataset_json_path xview_dataset/train.json \
  --val_dataset_json_path xview_dataset/val.json \
  --slice_size 300 \
  --overlap_ratio 0.2 \
  --ignore_negative_samples True \
  --min_area_ratio 0.1 \
  --output_train_dir xview_dataset/train_images_rgb_no_neg/ \
  --output_val_dir xview_dataset/val_images_rgb_no_neg/
```

# 4. Upload to s3 bucket to support distributed training

We will now upload our exported data to a publically accessible S3 bucket. This will enable for a large scale distributed experiment to have access to the dataset without installing the dataset on device. 
View these links to learn how to upload your dataset to an S3 bucket. Review the `S3Backend` class in `data.py`
* https://docs.determined.ai/latest/training/load-model-data.html#streaming-from-object-storage
* https://codingsight.com/upload-files-to-aws-s3-with-the-aws-cli/

Once you create an S3 bucket that is publically accessible, here are example commands to upload the preprocessed dataset to S3:
* `aws s3 cp --recursive xview_dataset/train_sliced_no_neg/   s3://determined-ai-xview-coco-dataset/train_sliced_no_neg`
* `aws s3 cp --recursive xview_dataset/val_sliced_no_neg/   s3://determined-ai-xview-coco-dataset/val_sliced_no_neg`


```python

```

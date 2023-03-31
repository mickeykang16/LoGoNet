# Getting Started
The dataset configs are located within [tools/cfgs/det_dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs/det_model_configs](../tools/cfgs/det_model_configs) for different datasets. 


## Dataset Preparation

Currently we provide the dataloader of KITTI, Waymo.

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
detection
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── al3d_det
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m al3d_det.datasets.kitti_dataset.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
## Dataset Preparation
Currently we provide the dataloader of Waymo dataset, and the supporting of more datasets are on the way.  

- please place the data folder based on the following structure:
	```
	detection
	├── data
	│   ├── waymo
	│   │   │── ImageSets
	│   │   │── raw_data
	│   │   │   │── segment-xxxxxxxx.tfrecord
	│   │   │   │── ....
	├── al3d_det
	├── tools
	```

- process waymo infos:
  ```
  cd detection
  python -m al3d_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml --func create_waymo_infos
  ```

- generate database for gt-sampling
  ```
  cd detection
  python -m al3d_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml --func create_waymo_database
  ```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data` to see how many records that have been processed): 

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 



## Training & Testing
### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
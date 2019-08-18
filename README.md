# hrnet-tf

## Overview 
This is a tensorflow implementation of [high-resolution representations for ImageNet classification](https://arxiv.org/abs/1904.04514). The network structure and training hyperparamters are kept the same as the offical [pytorch implementation](https://github.com/HRNet/HRNet-Image-Classification).

## Features of repo
* Low-level implementation of tensorflow
* Multiple GPU training via Horovod 
* Support configurable network for HRNet 
* Reproduce the on par accuracy of HRnet with its offical pytorch implementation.

## HRnet structure details
First, the four-resolution feature maps are fed into a bottleneck and the number of output channels are increased to 128, 256, 512, and 1024, respectively. Then, we downsample the high-resolution representations by a 2-strided 3x3 convolution outputting 256 channels and add them to the representations of the second-high-resolution representations. This process is repeated two times to get 1024 channels over the small resolution. Last, we transform 1024 channels to 2048 channels through a 1x1 convolution, followed by a global average pooling operation. The output 2048-dimensional representation is fed into the classifier.

## Accuracy of pretrained models 


## Installation
This repo is built on tensorflow 1.12 and Python 3.6
1. Install dependency 
```
pip install -r requirements.txt
```
2. [**Optional**] Follow [horovod installation instructions]() to install horovod to support multiple gpu training.

## Data preparision
Please follow [instructions](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) to converted imagenet dataset from images to tfrecords. This can accelerate the training speed significantly. After convertion, you will have tfrecords files under `data/tfrecords` as below
```
# training files
train-00000-of-01024
train-00001-of-01024
...

# validation files
validation-00000-of-00128
validation-00001-of-00128
...
```
## How to train and eval network 
1. Train network with one GPU for HRNet-W30
```
python top/train.py --net_cfg cfgs/w30_s4.cfg --data_path /path/to/tfrecords  
```

2.  If you want to resume training from saved checkpoint, set `resume_training` to enable resume training. 
```
python top/train.py --net_cfg cfgs/w30_s4.cfg --data_path /path/to/tfrecords --resume_training
```

3. Evaluate network. Make sure the checkpoint saved in `models`.
```
python top/train.py --net_cfg cfgs/w30_s4.cfg --data_path /path/to/tfrecords --eval_only
```

4. Training with multiple GPUs. Specify the number of gpus via `nb_gpus` and `extra_args` in  `./scripts/run_horovod.sh`.  For example, if you want to train HRNet-w30 by using 4 GPUs, the scripts would be like below

```
nb_gpus=4

extra_args='--net_cfg cfgs/w30_s4.cfg'

echo "multi-GPU training enabled"
mpirun -np ${nb_gpus} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python top/train.py --enbl_multi_gpu  
```


## Related Efforts 
1. Lot of code to build the dataset and training pipeline refer to [pocketflow](https://github.com/Tencent/PocketFlow)


## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao 
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
````
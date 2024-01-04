<div align="center">   
  
# Optimization of BEVFormer based on HoP module
</div>

</br>


# Abstract
In this work, we realize a state-of-the-art method named HoP on BEVFormer, which is published from [HoP](https://arxiv.org/abs/2304.00967). HoP is a plug-and-play approch which can be easily incorporated into BEV detection frameworks, including BEVFormer and BEVDet series. However, the author of HoP just publish the implementation of BEVDet. This work completed the implementation of BEVFormer. Detailed information about HoP and BEVFormer can be found in [HoP](https://github.com/Sense-X/HoP) and [BEVFormer](https://github.com/fundamentalvision/BEVFormer/tree/master).


# Getting Started

## Install

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n bf_hop python=3.8 -y
conda activate bf_hop
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**f. Install Detectron2 and Timm.**
```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


**g. Clone BEVFormer.**
```
git clone https://github.com/fool-githuber/BEV_with_HoP.git
```

**h. Prepare pretrained models.**
```shell
cd BEV_with_HoP/BEVFormer
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```
## Data Preparation

Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

## Run and Val

**train**
```
./tools/dist_train.sh ./projects/configs/bevformer_hop/bevformer_tiny_hop.py $nums_gpus
```
**test**
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_tiny_hop.py ./path/to/ckpts.pth $nums_gpus
```

# Model Zoo

| Backbone | Method | Lr Schd | NDS| mAP|memroy | Config |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: |
| R50 | BEVFormer-tiny | 24ep | 35.4|25.2 | 6500M |[config](projects/configs/bevformer/bevformer_tiny.py) |
| R50  | BEVFormer-tiny_hop | 24ep | 47.9|37.0 | 10500M |[config](projects/configs/bevformer/bevformer_small.py) |



# MHCLNet

- A Multi-Scale Pathological Context Learning Network for Breast Cancer Histopathology Image Classification

## Installation 

- To prepare the environment, please follow the following instructions.

  ```shell
  conda create --name openmmlab python=3.8 -y
  conda activate openmmlab
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
  git clone https://github.com/BaoLongS111/MHCLNet.git
  cd MHCLNet
  pip install -U openmim && mim install -e .
  ```

## Datasets

- The used datasets are provided in [BACH](https://iciar2018-challenge.grand-challenge.org/) and [BRACS](https://www.bracs.icar.cnr.it/). The train/test splits in both two datasets follow the official procedure. 

## Model

- We provide the original pretrained weights of Swin-L and the model weights of MHCLNet on the BACH and BRACS datasets. Please visit the following [link](https://pan.baidu.com/s/1nt9LOERcNLfcv-i3EdVPow?pwd=mlsf).

## Train

- The MHCLNet model file is located at `/MHCLNet/mmpretrain/models/backbones/modules/MHCLNet.py`. MHCLNet.py has been integrated into swin_transformer_mhclnet.py.

- If you want to train or test the model, please replace the contents of `/MHCLNet/mmpretrain/models/backbones/swin_transformer.py` with those in `/MHCLNet/mmpretrain/models/backbones/swin_transformer_mhclnet.py`. 

- Please make sure to back up the original SwinTransformer code in advance.
- Before training, please update the pretrained weight path and dataset path in the configuration file. The dataset should be organized in the ImageNet format.
  ```shell
  Dataset_ROOT_DIR/
    └──test/
        ├── ...
    └──train/
    	├── benign
    		├──0_0.png
    		├──0_1.png
    		├──0_2.png
    		├──...
    	├── malignant
    		├── ...
    		├── ...
    		├── ...
    	├── ...
  ```

- The model can be trained with the following command.

  ```shell
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  PORT=29209 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh ./swin_large_16xb64_in1k_BACH.py 2 --work-dir ./swinTransformer_result/bach/workdir
  ```

## Test

- The model can be tested with the following command,change the path below.

  ```shell
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  PORT=28756 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh ./swin_large_16xb64_in1k_BACH.py ./swinTransformer_result/bach/workdir/epoch_x.pth 2 --work-dir ./swinTransformer_result/bach/workdir/testx
  ```
- After downloading the fine-tuned MHCLNet model weights for the BACH and BRACS datasets, you can reproduce the results reported in the paper using the following evaluation command.
- ### BACH
  ```shell
  PORT=28756 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh ./swin_large_16xb64_in1k_BACH.py ./bach_96_38.pth 2 --work-dir ./swinTransformer_result/bach/workdir/test_bach
  ```
- ### BRACS
  ```shell
  PORT=28756 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh ./swin_large_16xb64_in1k_BRACS.py ./bracs_64_38.pth 2 --work-dir ./swinTransformer_result/bach/workdir/test_bracs
  ```
- If you would like to learn more about the training or testing command arguments, please visit this [link](https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/train.html).

## Acknowledgement

- This project is based on [MMPretrain](https://github.com/open-mmlab/mmpretrain). Thanks to the OpenMMLab team for their great work.

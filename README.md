# MHCLNet
- A Multi-Scale Pathological Context Learning Network for Breast Cancer Histopathology Image Classification
## Installation 
- To prepare the environment, please follow the following instructions.<br>
<code>conda create --name openmmlab python=3.8 -y</code><br>
<code>conda activate openmmlab</code><br>
<code>conda install pytorch torchvision -c pytorch</code> This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.<br>
<code>git clone https://github.com/BaoLongS111/MHCLNet.git</code><br>
<code>cd MLSFNet</code><br>
<code>pip install -U openmim && mim install -e .</code><br>
## Datasets
- The used datasets are provided in [BACH](https://iciar2018-challenge.grand-challenge.org/) and [BRACS](https://www.bracs.icar.cnr.it/). The train/test splits in both two datasets follow the official procedure. 
## Model
- We now provide the model weights in the following [link](https://pan.baidu.com/s/1nt9LOERcNLfcv-i3EdVPow?pwd=mlsf).
## Train
- The MHCLNet model file is located at `/MHCLNet/mmpretrain/models/backbones/modules/MHCLNet.py`. <br>
- If you want to train or test the model, please replace the contents of `/MHCLNet/mmpretrain/models/backbones/swin_transformer.py` with those in `/MHCLNet/mmpretrain/models/backbones/swin_transformer_mhclnet.py`. <br>
- Please make sure to back up the original SwinTransformer code in advance.<br>
- The model can be trained with the following command.<br>
<code>export CUBLAS_WORKSPACE_CONFIG=":4096:8"</code><br>
<code>CUDA_VISIBLE_DEVICES='0,1' bash tools/dist_train.sh /swin_large_16xb64_in1k_BACH.py 2 --seed 220 --deterministic</code><br>
- If you would like to learn more about the training or testing command arguments, please visit this [link](https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/train.html).
## Acknowledgement
- This project is based on [MMPretrain](https://github.com/open-mmlab/mmpretrain). Thanks to the OpenMMLab team for their great work.



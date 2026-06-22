# MHCLNet

- A Multi-Scale Pathological Context Learning Network for Breast Cancer Histopathology Image Classification

## Installation 

- To prepare the environment, please follow the following instructions.

  ```bash
  conda create --name vittt python=3.9 -y
  conda activate vittt
  conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
  pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.6.1 matplotlib==3.9.4 pillow==11.1.0
  pip install timm==0.4.12 einops==0.8.2 yacs==0.1.8
  ```

## Datasets

- The used datasets are provided in [BACH](https://iciar2018-challenge.grand-challenge.org/) and [BRACS](https://www.bracs.icar.cnr.it/). The train/test splits in BRACS dataset follow the official procedure. 

## Model

- We provide the original pretrained weights of H-VITTT-B and the model weights of MHCLNet on the BACH and BRACS datasets. Please visit the following [link](https://pan.baidu.com/s/1AQ9D7v7yHGD8Mk1lflW46Q?pwd=mhcl#list/path=%2F).

## Train

- The MHCLNet model file is located at `/MHCLNet/vittt/models/MHCLNet.py`. MHCLNet.py has been integrated into h_vittt_mhclnet.py.

- If you want to train or test the model, please replace the contents of `/MHCLNet/vittt/models/h_vittt.py` with those in `/MHCLNet/vittt/models/h_vittt_mhclnet.py`. 

- Please make sure to back up the original H-ViTTT-B code in advance.
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

  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./vittt/main_ema.py --cfg ./vittt/cfgs/h_vittt_b.yaml --data-path /home/bsj/data/BRACS_RoI_Normalized_512png2 --output /data/bsj/vittt/bracs/mhclnet --pretrained ./H-ViTTT-B-mesa.pth --batch-size 96 --freeze-backbone --amp --opts TRAIN.AUTO_RESUME False

  ```

## Test

- The model can be tested with the following command,change the path below.

  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./vittt/main_ema.py --cfg ./vittt/cfgs/h_vittt_b.yaml --data-path /home/bsj/data/BRACS_RoI_Normalized_512png2 --output /data/bsj/vittt/bracs/MHCLNet/test_result --eval --eval-split test --resume /data/bsj/vittt/bracs/MHCLNet/h_vittt_base/default/max_acc.pth --batch-size 96 --freeze-backbone --no-model-ema --opts TRAIN.AUTO_RESUME False
  ```
- After downloading the fine-tuned MHCLNet model weights for the BACH and BRACS datasets, you can reproduce the results reported in the paper using the following evaluation command.
  
  **BACH**
  
  Before executing the commands below, please modify the test weight path and the working output directory in vote_test_bach.py to the correct values.
  
  ```bash
  python ./vittt/vote_test_bach.py
  ```
  
  **BRACS**
  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 ./vittt/main_ema.py --cfg ./vittt/cfgs/h_vittt_b.yaml --data-path /dataset_path --output ./test_result --eval --eval-split test --resume ./bracs_64_04.pth --batch-size 96 --freeze-backbone --no-model-ema --opts TRAIN.AUTO_RESUME False

  ```
- If you would like to learn more about the training or testing command arguments, please visit this [link](https://github.com/LeapLabTHU/ViTTT).

## Acknowledgement

- This project is based on [ViTTT](https://github.com/LeapLabTHU/ViTTT). Thanks to the OpenMMLab team for their great work.

# SIIM-ACR Pneumothorax Segmentation

## First place solution 

### Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)
- Resnet50 from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)
- SCSEUnet (seresnext50) from \[[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)

### Main Features
- duplet/triplet scheme of validation/inference
- \[[combo loss\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py) combinations of BCE, dice and focal
    - (3,1,4)
    - (1,1,1)
    - (2,1,2)
- sliding sample rate
- best checkpoints averaging from each pipeline on inference
- horizontal flip TTA

### Install
```bash
pip install -r requirements.txt
```

### Pipeline launch example
Training:
```bash
python Train.py experiments/albunet_valid/train_config_part0.yaml
```
Inference:
```bash
python Inference.py experiments/albunet_valid/2nd_stage_inference.yaml
```
Submit:
```bash
python TripletSubmit.py experiments/albunet_valid/2nd_stage_submit.yaml
```

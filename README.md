# SIIM-ACR Pneumothorax Segmentation

## First place solution 

### Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)
- resnet50 from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)
- SCSEUnet (seresnext50) from \[[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)

### Main Features
- duplet/triplet scheme of validation/inference
- various combo loss combinations of BCE, focal and dice
- Sliding sample rate
- Best checkpoints averaging from each pipeline

### Install
```bash
pip install -r requirements.txt
```

### Pipeline launch example
Training:
```bash
python Train.py experiments/albunet_valid/train_config_part0.yaml
```
```bash
Inference: python Inference.py experiments/albunet_valid/2nd_stage_inference.yaml
```
```bash
Submit: python TripletSubmit.py experiments/albunet_valid/2nd_stage_submit.yaml
```

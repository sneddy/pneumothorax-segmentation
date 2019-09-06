# SIIM-ACR Pneumothorax Segmentation

## First place solution 

### Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)
- Resnet50 from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)
- SCSEUnet (seresnext50) from \[[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)

### Main Features
- **Triplet scheme of inference**
Instead of classification models for pneumathorax/non-pneumathorax images, I used two different thresholds: 
    - first one for mask binarization and transform in 
    - second threshold is maximum allowed number of pixels with value greater than the first threshold
Those images that didn't pass this pair of thresholds were counted non-pneumathorax images. 
For other images 
    
- Used \[[combo loss\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py) combinations of BCE, dice and focal
    - (3,1,4)
    - (1,1,1)
    - (2,1,2)
- **sliding sample rate**

Let's name portion of pneumathorax images as sample rate.

Main idea: control this portion using sampler of torch dataset. 

On each epoch my sampler get all images from dataset with pneumathorax and sample some from non-pneumathorax according to this sample rate. During train process we reduce this parameter from 0.8 on start to 0.4 in the end.

Large sample rate at the beginning provides a quick start of learning process, whereas a small sample rate provides better convergence of neural network weights to the initial distribution of pneumathorax/non-pneumathorax images.


- best checkpoints averaging from each pipeline on inference
- horizontal flip TTA

### Data Preparation

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

### Best experiments:
- AlbunetPublic - best model for Public Leaderboard
- AlbunetValid - best resnet34 model on validation
- seunet - best seresnext50 model on validation
- resnet50 - best resnet50 model on validation


### Final Submission
My best model for Public Leaderboard was AlbunetPublic (PL: 0.8871), and score of all ensembling models was worse.
But I suspected overfitting for this model therefore both final submissions were ensembles.

Private Leaderboard:
- 0.8679
- 0.8641



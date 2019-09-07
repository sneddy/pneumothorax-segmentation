# SIIM-ACR Pneumothorax Segmentation

# First place solution 

## Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)
- Resnet50 from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)
- SCSEUnet (seresnext50) from \[[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)

## Main Features
### Triplet scheme of inference
Let our segmentation model output some mask with probabilities of pneumothorax pixels. I'm going to name this mask as a basic sigmoid mask. I used triplet of different thresholds: *(top_score_threshold, min_contour_area, bottom_score_threshold)*

The decision rule is based on a doublet *(top_score_threshold, min_contour_area)*. I used it instead of using the classification of pneumothorax/non-pneumothorax.
- *top_score_threshold* is simple binarization threshold and transform basic sigmoid mask into a discrete mask of zeros and ones.
- *min_contour_area* is the maximum allowed number of pixels with a value greater than top_score_threshold

Those images that didn't pass this doublet of thresholds were counted non-pneumothorax images.

For the remaining pneumothorax images, we binarize basic sigmoid mask using *bottom_score_threshold* (another binariztion threshold).

The simplified version of this scheme:
```python
classification_mask = predicted > top_score_threshold
mask = predicted.copy()
mask[classification_mask.sum(axis=(1,2,3)) < min_contour_area, :,:,:] = np.zeros_like(predicted[0])
mask = mask > bot_score_threshold
return mask
```

### Search best triplet thresholds during validation 
- Best triplet on validation: (0.75, 2000, 0.3).
- Best triplet on Public Leaderboard: (0.7, 600, 0.3)

### Combo loss
Used \[[combo loss\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py) combinations of BCE, dice and focal. In the best experiments the weights of (BCE, dice, focal), that I used were:
- (3,1,4);
- (1,1,1);
- (2,1,2) respectively.
 
### Sliding sample rate
Let's name portion of pneumathorax images as sample rate.

The main idea is control this portion using sampler of torch dataset. 

On each epoch, my sampler gets all images from a dataset with pneumothorax and sample some from non-pneumothorax according to this sample rate. During train process, we reduce this parameter from 0.8 on start to 0.4 in the end.

Large sample rate at the beginning provides a quick start of the learning process, whereas a small sample rate at the end provides better convergence of neural network weights to the initial distribution of pneumothorax/non-pneumothorax images.

### Learning Process recipes
I can't provide fully reproducible solution because  during learning process I was uptrain my models **A LOT**. But looking back for formalization of my experiments I can highlight 4 different parts:
- **part 0** - train for 10-12 epoches from pretrained model with large learning rate (about 1e-3 or 1e-4), large sample rate (0.8) and ReduceLROnPlateau scheduller. Model can be pretrained on imagenet or on our dataset with lower resolution (512x512).  Goal of this part: quickly get a good enough model with validation score about 0.835. 
- **part 1** - uptrain best model from previous step with normal learning rate (~1e-5), large sample rate (0.6) and CosineAnnealingLR or CosineAnnealingWarmRestarts scheduler. Repeat until best convergence.
- **part 2** - uptrain best model from previous step with normal learning rate (~1e-5), small sample rate (0.4) and CosineAnnealingLR or CosineAnnealingWarmRestarts scheduler. Repeat until best convergence.
- **second_stage** - simple uptrain with relatively small learning rate(1e-5 or 1e-6), small sample rate (0.5) and CosineAnnealingLR or CosineAnnealingWarmRestarts scheduler.

All these parts are presented in the corresponding experiment folder
### Augmentations
Used following transforms from \[[albumentations\]](https://github.com/albu/albumentations)
```python
albu.Compose([
    albu.HorizontalFlip(),
    albu.OneOf([
        albu.RandomContrast(),
        albu.RandomGamma(),
        albu.RandomBrightness(),
        ], p=0.3),
    albu.OneOf([
        albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        albu.GridDistortion(),
        albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    albu.ShiftScaleRotate(),
    albu.Resize(img_size,img_size,always_apply=True),
])
```
### Uptrain from lower resolution
All experiments (except resnet50) uptrained on size 1024x1024 after 512x512 with frozen encoder on early epoches.  

### Second stage uptrain
All choosen experiments was uptrained on second stage data

### Checkpoints averaging
Top3 checkpoints averaging from each pipeline on inference

### Horizontal flip TTA

## Install
```bash
pip install -r requirements.txt
```

## Data Preparation
here will be guide for transforming input data to my format

## Pipeline launch example
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

## Best experiments:
- albunet_public - best model for Public Leaderboard
- albunet_valid - best resnet34 model on validation
- seunet - best seresnext50 model on validation
- resnet50 - best resnet50 model on validation

![picture alt](https://github.com/sneddy/kaggle-pneumathorax/blob/master/dashboard.png)


## Final Submission
My best model for Public Leaderboard was albunet_public (PL: 0.8871), and score of all ensembling models was worse.
But I suspected overfitting for this model therefore both final submissions were ensembles.

- First ensemble believed in Public Leaderboard scores more and used more "weak" triplet thresholds.
- Second ensemble believed in the validation scores more, but used more "strict" triplet thresholds.

### Private Leaderboard:
- 0.8679
- 0.8641

I suspect that the best solution would be ensemble believed in the validation scores more, but used more "weak" triplet thresholds.



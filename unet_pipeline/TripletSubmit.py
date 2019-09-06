import argparse
import pickle
from tqdm import tqdm
from pathlib import Path

import cv2

import numpy as np
import pandas as pd
from collections import defaultdict

from utils.mask_functions import mask2rle
from utils.helpers import load_yaml

def argparser():
    parser = argparse.ArgumentParser(description='Pneumatorax pipeline')
    parser.add_argument('cfg', type=str, help='experiment name')
    return parser.parse_args()

def extract_largest(mask, n_objects):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]]
    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours[:n_objects],
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def remove_smallest(mask, min_contour_area):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours,
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def apply_thresholds(mask, n_objects, area_threshold, top_score_threshold, 
                     bottom_score_threshold, leak_score_threshold, use_contours, min_contour_area):
    if n_objects == 1:
        crazy_mask = (mask > top_score_threshold).astype(np.uint8)
        if crazy_mask.sum() < area_threshold: 
            return -1
        mask = (mask > bottom_score_threshold).astype(np.uint8)
    else:
        mask = (mask > leak_score_threshold).astype(np.uint8)

    if min_contour_area > 0:
        choosen = remove_smallest(mask, min_contour_area)
    elif use_contours:
        choosen = extract_largest(mask, n_objects)
    else:
        choosen = mask * 255

    if mask.shape[0] == 1024:
        reshaped_mask = choosen
    else:
        reshaped_mask = cv2.resize(
            choosen,
            dsize=(1024, 1024),
            interpolation=cv2.INTER_LINEAR
        )
    reshaped_mask = (reshaped_mask > 127).astype(int) * 255
    return mask2rle(reshaped_mask.T, 1024, 1024)

def build_rle_dict(mask_dict, n_objects_dict,  
                   area_threshold, top_score_threshold,
                   bottom_score_threshold,
                   leak_score_threshold, 
                   use_contours, min_contour_area):  
    rle_dict = {}
    for name, mask in tqdm(mask_dict.items()):
        n_objects = n_objects_dict.get(name, 0)
        if n_objects == 0:
            continue
        rle_dict[name] = apply_thresholds(
            mask, n_objects, 
            area_threshold, top_score_threshold,
            bottom_score_threshold,
            leak_score_threshold, 
            use_contours, min_contour_area
        )
    return rle_dict

def buid_submission(rle_dict, sample_sub):
    sub = pd.DataFrame.from_dict([rle_dict]).T.reset_index()
    sub.columns = sample_sub.columns
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = -1
    return sub

def load_mask_dict(cfg):
    reshape_mode = cfg.get('RESHAPE_MODE', False)
    if 'MASK_DICT' in cfg:
        result_path = Path(cfg['MASK_DICT'])
        with open(result_path, 'rb') as handle:
            mask_dict = pickle.load(handle)
        return mask_dict
    if 'RESULT_WEIGHTS' in cfg:
        result_weights = cfg['RESULT_WEIGHTS']
        mask_dict = defaultdict(int)
        for result_path, weight in result_weights.items():
            print(result_path, weight)
            with open(Path(result_path), 'rb') as handle:
                current_mask_dict = pickle.load(handle)
                for name, mask in current_mask_dict.items():
                    if reshape_mode and mask.shape[0] != 1024:
                        mask = cv2.resize(
                            mask,
                            dsize=(1024, 1024), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    #crazy_mask = (mask > 0.75).astype(np.uint8)
                    #if crazy_mask.sum() < 1000:
                    #  mask = np.zeros_like(mask)
                    mask_dict[name] = mask_dict[name] + mask * weight
        return mask_dict


def main():
    args = argparser()
    config_path = Path(args.cfg.strip("/"))
    sub_config = load_yaml(config_path)
    print(sub_config)
    
    sample_sub = pd.read_csv(sub_config['SAMPLE_SUB'])
    n_objects_dict = sample_sub.ImageId.value_counts().to_dict()
    
    print('start loading mask results....')
    mask_dict = load_mask_dict(sub_config)
    
    use_contours = sub_config['USECONTOURS']
    min_contour_area = sub_config.get('MIN_CONTOUR_AREA', 0)

    area_threshold = sub_config['AREA_THRESHOLD']
    top_score_threshold = sub_config['TOP_SCORE_THRESHOLD']
    bottom_score_threshold = sub_config['BOTTOM_SCORE_THRESHOLD']
    if sub_config['USELEAK']:
        leak_score_threshold = sub_config['LEAK_SCORE_THRESHOLD']
    else:
        leak_score_threshold = bottom_score_threshold

    rle_dict = build_rle_dict(
        mask_dict, n_objects_dict, area_threshold,
        top_score_threshold, bottom_score_threshold,
        leak_score_threshold, use_contours, min_contour_area
    )
    sub = buid_submission(rle_dict, sample_sub)
    print((sub.EncodedPixels != -1).sum())
    print(sub.head())
    
    sub_file = Path(sub_config['SUB_FILE'])
    sub.to_csv(sub_file, index=False)

if __name__ == "__main__":
    main()

import argparse
from tqdm import tqdm
import os
import importlib
from pathlib import Path
import pickle

import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
import albumentations as albu 
import torch

from Pneumadataset import PneumothoraxDataset
from utils.helpers import load_yaml, init_seed, init_logger


def argparser():
    parser = argparse.ArgumentParser(description='Pneumatorax pipeline')
    parser.add_argument('cfg', type=str, help='experiment name')
    return parser.parse_args()

def build_checkpoints_list(cfg):
    pipeline_path = Path(cfg['CHECKPOINTS']['PIPELINE_PATH'])
    pipeline_name = cfg['CHECKPOINTS']['PIPELINE_NAME']

    checkpoints_list = []
    if cfg.get('SUBMIT_BEST', False):
        best_checkpoints_folder = Path(
            pipeline_path, 
            cfg['CHECKPOINTS']['BEST_FOLDER']
        )

        usefolds = cfg['USEFOLDS']
        for fold_id in usefolds:
            filename = '{}_fold{}.pth'.format(pipeline_name, fold_id)
            checkpoints_list.append(Path(best_checkpoints_folder, filename))
    else:
        folds_dict = cfg['SELECTED_CHECKPOINTS']
        for folder_name, epoch_list in folds_dict.items():
            checkpoint_folder = Path(
                pipeline_path,
                cfg['CHECKPOINTS']['FULL_FOLDER'],
                folder_name,
            )
            for epoch in epoch_list:
                checkpoint_path = Path(
                    checkpoint_folder,
                    '{}_{}_epoch{}.pth'.format(pipeline_name, folder_name, epoch)
                )
                checkpoints_list.append(checkpoint_path)
    return checkpoints_list


def inference_image(model, images, device):
    images = images.to(device)
    predicted = model(images)
    masks = torch.sigmoid(predicted) 
    masks = masks.squeeze(1).cpu().detach().numpy()
    return masks


def inference_model(model, loader, device, use_flip):
    mask_dict = {}
    for image_ids, images in tqdm(loader):
        masks = inference_image(model, images, device)
        if use_flip:
            flipped_imgs = torch.flip(images, dims=(3,))
            flipped_masks = inference_image(model, flipped_imgs, device)
            flipped_masks = np.flip(flipped_masks, axis=2)
            masks = (masks + flipped_masks) / 2
        for name, mask in zip(image_ids, masks):
            mask_dict[name] = mask.astype(np.float32)
    return mask_dict
    

def main():
    args = argparser()
    config_path = Path(args.cfg.strip("/"))
    experiment_folder = config_path.parents[0]
    inference_config = load_yaml(config_path)
    print(inference_config)
    
    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']
    
    module = importlib.import_module(inference_config['MODEL']['PY'])
    model_class = getattr(module, inference_config['MODEL']['CLASS'])
    model = model_class(**inference_config['MODEL'].get('ARGS', None)).to(device)
    model.eval()

    num_workers = inference_config['NUM_WORKERS']
    transform = albu.load(inference_config['TEST_TRANSFORMS']) 
    dataset_folder = inference_config['DATA_DIRECTORY'] 
    dataset = PneumothoraxDataset(
        data_folder=dataset_folder, mode='test', 
        transform=transform,
    )
    dataloader =  DataLoader(
        dataset=dataset, batch_size=batch_size, 
        num_workers=num_workers, shuffle=False
    )

    use_flip = inference_config['FLIP']
    checkpoints_list = build_checkpoints_list(inference_config)
  
    mask_dict = defaultdict(int)
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        current_mask_dict = inference_model(model, dataloader, device, use_flip)
        for name, mask in current_mask_dict.items():
            mask_dict[name] = (mask_dict[name] * pred_idx + mask) / (pred_idx + 1)

    if 'RESULT_FOLDER' in inference_config:
        result_path = Path(inference_config['RESULT_FOLDER'], inference_config['RESULT'])
    else:
        result_path = Path(experiment_folder, inference_config['RESULT'])

    with open(result_path, 'wb') as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    main()
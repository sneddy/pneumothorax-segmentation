import argparse
import logging

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import albumentations as albu 
import torch

import importlib
import functools
from tqdm import tqdm
import os
from pathlib import Path

from Pneumadataset import PneumothoraxDataset, PneumoSampler
from Learning import Learning
from utils.helpers import load_yaml, init_seed, init_logger
# from Evaluation import apply_deep_thresholds, search_deep_thresholds, dice_round_fn, search_thresholds


def argparser():
    parser = argparse.ArgumentParser(description='Pneumatorax pipeline')
    parser.add_argument('train_cfg', type=str, help='train config path')
    return parser.parse_args()

    
def train_fold(
    train_config, experiment_folder, pipeline_name, log_dir, fold_id,
    train_dataloader, valid_dataloader, binarizer_fn, eval_fn):
    
    fold_logger = init_logger(log_dir, 'train_fold_{}.log'.format(fold_id))

    best_checkpoint_folder = Path(experiment_folder, train_config['CHECKPOINTS']['BEST_FOLDER'])
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config['CHECKPOINTS']['FULL_FOLDER'],
        'fold{}'.format(fold_id)
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config['CHECKPOINTS']['TOPK']

    calculation_name = '{}_fold{}'.format(pipeline_name, fold_id)
    
    device = train_config['DEVICE']
    
    module = importlib.import_module(train_config['MODEL']['PY'])
    model_class = getattr(module, train_config['MODEL']['CLASS'])
    model = model_class(**train_config['MODEL']['ARGS'])

    pretrained_model_config = train_config['MODEL'].get('PRETRAINED', False)
    if pretrained_model_config: 
        loaded_pipeline_name = pretrained_model_config['PIPELINE_NAME']
        pretrained_model_path = Path(
            pretrained_model_config['PIPELINE_PATH'], 
            pretrained_model_config['CHECKPOINTS_FOLDER'],
            '{}_fold{}.pth'.format(loaded_pipeline_name, fold_id)
        ) 
        if pretrained_model_path.is_file():
            model.load_state_dict(torch.load(pretrained_model_path))
            fold_logger.info('load model from {}'.format(pretrained_model_path)) 

    if len(train_config['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)
    
    module = importlib.import_module(train_config['CRITERION']['PY'])
    loss_class = getattr(module, train_config['CRITERION']['CLASS'])
    loss_fn = loss_class(**train_config['CRITERION']['ARGS'])
    
    optimizer_class = getattr(torch.optim, train_config['OPTIMIZER']['CLASS'])
    optimizer = optimizer_class(model.parameters(), **train_config['OPTIMIZER']['ARGS'])
    scheduler_class = getattr(torch.optim.lr_scheduler, train_config['SCHEDULER']['CLASS'])
    scheduler = scheduler_class(optimizer, **train_config['SCHEDULER']['ARGS'])
    
    n_epoches = train_config['EPOCHES']
    grad_clip = train_config['GRADIENT_CLIPPING']
    grad_accum = train_config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['EARLY_STOPPING']
    validation_frequency = train_config.get('VALIDATION_FREQUENCY', 1)
    
    freeze_model = train_config['MODEL']['FREEZE']
    
    Learning(
        optimizer,
        binarizer_fn,
        loss_fn,
        eval_fn,
        device,
        n_epoches,
        scheduler,
        freeze_model,
        grad_clip,
        grad_accum,
        early_stopping,
        validation_frequency,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        fold_logger
    ).run_train(model,train_dataloader,valid_dataloader)

def main():
    args = argparser()
    config_folder = Path(args.train_cfg.strip("/"))
    experiment_folder = config_folder.parents[0]

    train_config = load_yaml(config_folder)

    log_dir = Path(experiment_folder, train_config['LOGGER_DIR'])
    log_dir.mkdir(exist_ok=True, parents=True)

    main_logger = init_logger(log_dir, 'train_main.log')

    seed = train_config['SEED']
    init_seed(seed)
    main_logger.info(train_config)

    if "DEVICE_LIST" in train_config:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, train_config["DEVICE_LIST"]))

    pipeline_name = train_config['PIPELINE_NAME']
    dataset_folder = train_config['DATA_DIRECTORY'] 

    train_transform = albu.load(train_config['TRAIN_TRANSFORMS']) 
    valid_transform = albu.load(train_config['VALID_TRANSFORMS'])

    non_empty_mask_proba = train_config.get('NON_EMPTY_MASK_PROBA', 0)
    use_sampler = train_config['USE_SAMPLER']

    dataset_folder = train_config['DATA_DIRECTORY'] 
    folds_distr_path = train_config['FOLD']['FILE'] 

    num_workers = train_config['WORKERS'] 
    batch_size = train_config['BATCH_SIZE'] 
    n_folds = train_config['FOLD']['NUMBER'] 

    usefolds = map(str, train_config['FOLD']['USEFOLDS'])
    # local_metric_fn, global_metric_fn = init_eval_fns(train_config)

    binarizer_module = importlib.import_module(train_config['MASK_BINARIZER']['PY'])
    binarizer_class = getattr(binarizer_module, train_config['MASK_BINARIZER']['CLASS'])
    binarizer_fn = binarizer_class(**train_config['MASK_BINARIZER']['ARGS'])

    eval_module = importlib.import_module(train_config['EVALUATION_METRIC']['PY'])
    eval_fn = getattr(eval_module, train_config['EVALUATION_METRIC']['CLASS'])
    eval_fn = functools.partial(eval_fn, **train_config['EVALUATION_METRIC']['ARGS'])

    for fold_id in usefolds:
        main_logger.info('Start training of {} fold....'.format(fold_id))

        train_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='train', 
            transform=train_transform, fold_index=fold_id,
            folds_distr_path=folds_distr_path,
        )
        train_sampler = PneumoSampler(folds_distr_path, fold_id, non_empty_mask_proba)
        if use_sampler:
            train_dataloader =  DataLoader(
                dataset=train_dataset, batch_size=batch_size,   
                num_workers=num_workers, sampler=train_sampler
            )
        else:
            train_dataloader =  DataLoader(
                dataset=train_dataset, batch_size=batch_size,   
                num_workers=num_workers, shuffle=True
            )

        valid_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='val', 
            transform=valid_transform, fold_index=str(fold_id),
            folds_distr_path=folds_distr_path,
        )
        valid_dataloader =  DataLoader(
            dataset=valid_dataset, batch_size=batch_size, 
            num_workers=num_workers, shuffle=False
        )

        train_fold(
            train_config, experiment_folder, pipeline_name, log_dir, fold_id,
            train_dataloader, valid_dataloader,
            binarizer_fn, eval_fn
        )


if __name__ == "__main__":
    main()

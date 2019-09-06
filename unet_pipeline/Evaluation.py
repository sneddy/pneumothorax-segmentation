import numpy as np
#import torch
#from losses import dice_round
from tqdm import tqdm
from joblib import Parallel, delayed


def dice_loss_fn(predicted, ground_truth):
    eps = 1e-4
    batch_size = predicted.shape[0]

    predicted = predicted.reshape(batch_size, -1).astype(np.bool)
    ground_truth = ground_truth.reshape(batch_size, -1).astype(np.bool)

    intersection = np.logical_and(predicted, ground_truth).sum(axis=1)
    union = predicted.sum(axis=1) + ground_truth.sum(axis=1) + eps
    loss = (2. * intersection + eps) / union
    return loss.mean()

def dice_round_fn(predicted, ground_truth, score_threshold=0.5, area_threshold=0):
    mask = predicted > score_threshold
    mask[mask.sum(axis=(1,2,3)) < area_threshold, :,:,:] = np.zeros_like(mask[0])
    return dice_loss_fn(mask, ground_truth)

def search_thresholds(eval_list, thr_list, area_list, n_search_workers):
    best_score = 0
    best_thr = -1
    best_area = -1

    progress_bar = tqdm(thr_list)
    for thr in progress_bar:
        for area in area_list:
            score_list = Parallel(n_jobs=n_search_workers)(delayed(dice_round_fn)(
            	probas, labels, thr, area) for probas, labels in eval_list)
            final_score = np.mean(score_list)
            if final_score > best_score:
                best_score = final_score
                best_thr = thr
                best_area = area
            progress_bar.set_description('Best score: {:.3}'.format(best_score))
    return best_thr, best_area, best_score


def apply_deep_thresholds(predicted, ground_truth, top_score_threshold=0.5, bot_score_threshold=0.4, area_threshold=0):
    classification_mask = predicted > top_score_threshold
    mask = predicted.copy()
    mask[classification_mask.sum(axis=(1,2,3)) < area_threshold, :,:,:] = np.zeros_like(predicted[0])
    mask = mask > bot_score_threshold
    return dice_loss_fn(mask, ground_truth)

def search_deep_thresholds(eval_list, triplets_list, n_search_workers):
    best_score = 0
    best_thr = -1
    best_area = -1

    progress_bar = tqdm(triplets_list)

    for top_thr, area_thr, bot_thr in progress_bar:
        score_list = Parallel(n_jobs=n_search_workers)(delayed(apply_deep_thresholds)(
        	probas, labels, top_thr, bot_thr, area_thr) for probas, labels in eval_list)
        final_score = np.mean(score_list)
#        print(top_thr, area_thr, bot_thr, final_score)
        if final_score > best_score:
            best_score = final_score
            best_triplets = top_thr, area_thr, bot_thr
        progress_bar.set_description('Best score: {:.4}'.format(best_score))
    return best_score, best_triplets

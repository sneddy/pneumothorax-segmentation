import numpy as np 
import pandas as pd 

import cv2
import os
from pathlib import Path
from tqdm import tqdm

from joblib import Parallel, delayed

import argparse
from mask_functions import rle2mask


def argparser():
	parser = argparse.ArgumentParser(description='Prepare png dataset for pneumatorax')
	parser.add_argument('-sub_path', default='../input/train-rle.csv', type=str, nargs='?', help='path for rle csv file')
	parser.add_argument('-out_path', default='../dataset', type=str, nargs='?', help='path for saving masks')
	parser.add_argument('-n_threads', default=4, type=int, nargs='?', help='number of using threads')
	return parser.parse_args()

def process_record(record, out_dir):
	image_id = record['ImageId'] + '.png'
	current_rle = record['EncodedPixels'].strip()
	mask = rle2mask(current_rle, 1024, 1024).T
	out_file = Path(out_dir, image_id) 
	cv2.imwrite(str(out_file), mask) 

def main():
	args = argparser()
	
	sub = pd.read_csv(args.sub_path)
	out_dir = Path(args.out_path)
	n_threads = args.n_threads
	out_dir.mkdir(exist_ok=True)

	Parallel(n_jobs=n_threads, backend='threading')(delayed(process_record)(
		record, out_dir) for _, record in tqdm(sub.iterrows(), total=sub.shape[0]))

if __name__ == '__main__':
	main()
from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from utils.core_utils import train, test

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler

def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    
    args.omic_sizes = [144, 521, 668, 680, 2365, 989]

    if args.op_mode == 'test':
        test_path = os.path.join(args.data_root_dir, 'test/')
        datasets = [[test_path]]
        test(datasets, args)
        return
    
    train_path = os.path.join(args.data_root_dir, 'train/')
    val_path = os.path.join(args.data_root_dir, 'val/')
    test_path = os.path.join(args.data_root_dir, 'test/')
    datasets = ([train_path, val_path], [test_path])

    train(datasets, args)

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Genomic data generation from WSI on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='Data/TCGA_KIRC/processed_data/', help='Data directory')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--results_dir',     type=str, default='Code_KIRC/P2G/results/', help='Results directory (Default: ./results)')
parser.add_argument('--weight_path', type=str, default='Code_KIRC/P2G/results/1000T_8gc_mms/w49.pt', help='Path of weight to initialize with.')
parser.add_argument('--op_mode', type=str, default='train', help='train or test')

### Model Parameters.
parser.add_argument('--T', type=int, default=1000, help='Number of diffusion timesteps. (default: 1000)')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='big', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='big', help='Network size of SNN model')

### Optimizer Parameters + Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=8, help='Gradient Accumulation Step. (default: 32)')
parser.add_argument('--start_epoch',      type=int, default=0, help='Start epoch for train (default: 0)')
parser.add_argument('--max_epochs',      type=int, default=70, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--save_epoch',      type=int, default=1, help='epochs to save (default: 5)')
parser.add_argument('--lr',				 type=float, default=1e-4, help='Learning rate (default: 0.0001)')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir += str(args.T)+'T_'+str(args.gc)+'/'
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)       

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))

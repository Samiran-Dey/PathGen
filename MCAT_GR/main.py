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
from utils.core_utils import train, test, calibration
from utils.utils import get_custom_exp_code

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

    train_path = os.path.join(args.data_root_dir, 'train/')
    val_path = os.path.join(args.data_root_dir, 'val/')
    cal_path = os.path.join(args.data_root_dir, 'cal/')
    test_path = os.path.join(args.data_root_dir, 'test/')

    if args.op_mode == 'test':
        datasets = [[test_path]]
        test(datasets, args)
        return

    if args.op_mode == 'calibrate':
        datasets = [[cal_path],[test_path]]
        calibration(datasets, args)
        return


    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex, latest_val_auc = [], []
    folds = np.arange(start, end)
    
    

    print('Lambda survival: ', args.loss_alpha)
    
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        datasets = ([train_path, val_path], [test_path])

        cindex_latest, auc_latest = train(datasets, i, args)
        latest_val_cindex.append(cindex_latest)
        latest_val_auc.append(auc_latest)

        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex, 'val_auc': latest_val_auc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

real_ratio = 1



loss_coef = 0.3
### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='/Data/TCGA_KIRC/processed_data/', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=1, help='Number of folds (default: 1)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='Code_KIRC/MCAT_exp/results/', help='Results directory (Default: ./results)')
parser.add_argument('--best_weight_path',   type=str, default='Code_KIRC/MCAT_exp/results/lc'+str(loss_coef)+'/0_best_weights.pt', help='Path of weight to initialize model with.')
parser.add_argument('--test_syn_path',   type=str, default='Code_KIRC/P2G/results/1000T_8_mms/epoch49pred_test_sm.pt', help='Path of sythesized genomic data.')
parser.add_argument('--op_mode', type=str, choices = ['train','test', 'calibrate'], default='train', help='Type of execution.')
parser.add_argument('--data_type', type=str, choices=['real','syn','merged','fused', 'noise'], default='syn', help='Load synthesized genomic data or real or fused.')
parser.add_argument('--test_type', type=str, choices=['overall','distributed'], default='overall', help='Specify the test type.')

### Model Parameters.
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='concat', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_omic', type=str, default='big', help='Network size of SNN model')
parser.add_argument('--n_timebin', type=int, default=4, help='Number of time bins.')
parser.add_argument('--n_grade', type=int, default=4, help='Number of grades.')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=16, help='Gradient Accumulation Step. (default: 32)')
parser.add_argument('--max_epochs',      type=int, default=25, help='Maximum number of epochs to train (default: 25)')
parser.add_argument('--save_epoch',      type=int, default=1, help='epochs to save (default: 5)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--loss_alpha',      type=float, default=loss_coef, help='weight between survival and gradation tasks (Default 0.5)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.alpha_syn = 1
args.real_ratio = real_ratio

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

args.results_dir += 'lc'+str(args.loss_alpha)+'/test_result/'
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)   


if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))


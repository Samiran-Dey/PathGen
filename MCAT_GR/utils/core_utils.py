
from argparse import Namespace
from collections import OrderedDict
import os
import pickle 
import time
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from models.model_genomic import SNN
from models.model_coattn_genomic import MCAT_Surv
from utils.utils import *
from utils.coattn_train_utils import *
from utils.test_utils import *
from utils.calibration_utils import *
from utils.data_utils import create_dataset

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss



def train(datasets: tuple, cur: int, args: Namespace):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    print('\nTraining Fold {}!'.format(cur))

    writer = os.path.join(args.results_dir, 'loss_log.txt')
    with open(writer, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('====== Training Loss (%s)  ======\n' % now)

    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'ce_surv':
        loss_fn_surv = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn_surv = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn_surv = CoxSurvLoss()
    else:
        raise NotImplementedError


    loss_fn_grad = torch.nn.BCEWithLogitsLoss().to(device)

    print('Done!')
    
    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_timebin, 'n_grad': args.n_grade}
    model = MCAT_Surv(**model_dict)
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = create_dataset(args, datasets[0], batch_size=args.batch_size)
    val_loader = create_dataset(args, datasets[1], batch_size=args.batch_size)
    
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    best_score_ci, best_score_auc, best_ci, best_auc = 0, 0, 0, 0

    for epoch in range(args.max_epochs):
        train_loop_survival_coattn(args, cur, epoch, model, train_loader, optimizer, args.n_timebin, writer, loss_fn_surv, loss_fn_grad, args.loss_alpha)
        stop, c_index, kappa, roc_auc = validate_survival_coattn(args, cur, epoch, model, val_loader, args.n_timebin, early_stopping, writer, loss_fn_surv, loss_fn_grad, args.results_dir)
        
        if c_index >= best_score_ci and roc_auc >= best_score_auc:
            torch.save(model.state_dict(), args.results_dir+str(cur)+'_best_weights.pt')
            msg='CI improved by %.5f and AUC improved by %.5f at epoch %d. Best weights saved.'%((c_index - best_score_ci), (roc_auc-best_score_auc), epoch)
            print(msg)
            with open(writer, "a") as log_file:
                log_file.write('%s\n' % msg)
            best_score_ci = c_index
            best_score_auc = roc_auc
        
        if c_index >= best_ci:
            torch.save(model.state_dict(), args.results_dir+str(cur)+'_best_weights_ci.pt')
            msg='CI improved by %.5f at epoch %d. Best weights saved.'%((c_index - best_ci), epoch)
            print(msg)
            with open(writer, "a") as log_file:
                log_file.write('%s\n' % msg)
            best_ci = c_index
        
        if roc_auc >= best_auc:
            torch.save(model.state_dict(), args.results_dir+str(cur)+'_best_weights_auc.pt')
            msg='AUC improved by %.5f at epoch %d. Best weights saved.'%((roc_auc - best_auc), epoch)
            print(msg)
            with open(writer, "a") as log_file:
                log_file.write('%s\n' % msg)
            best_auc = roc_auc       

    return best_ci, best_auc


def test(datasets: tuple, args: Namespace):
    
    print('loss coefficient: ', args.loss_alpha)
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    writer_name = args.test_type+'_test_log.txt'
    writer = os.path.join(args.results_dir, writer_name)
    with open(writer, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('====== Test scores (%s) - %s  ====== \n' % (now, args.data_type))

    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_timebin, 'n_grad': args.n_grade}
    model_best = MCAT_Surv(**model_dict)
    model_best.load_state_dict(torch.load(args.best_weight_path, weights_only=False))
    model_best.to(device)

    print('Done!')

    print('\nInit Loaders...', end=' ')
    test_loader = create_dataset(args, datasets[0], batch_size=args.batch_size)

    if args.test_type == 'overall':
        print('Producing overall test results')
        test_surv_grad(args, model_best, test_loader, writer, args.results_dir)
    elif args.test_type == 'distributed':
        print('Producing distributed test results')
        test_distributed(args, model_best, test_loader, writer, args.results_dir)


def calibrate_now(args, datasets, model, writer, data_filter, category):
    print('\nInit Loaders ', category, '...', end=' ')
    try:
        cal_loader = create_dataset(args, datasets[0], op_mode = 'train', batch_size=args.batch_size, data_filter=data_filter)
        test_loader = create_dataset(args, datasets[1], op_mode = 'test', batch_size=args.batch_size, data_filter=data_filter)
        loader = [cal_loader, test_loader]

        result_dct = test_calibrate(args, model, loader, args.n_timebin, writer, category=category)
        return result_dct
    except:
        return None

def calibration(datasets: tuple, args: Namespace):
    
    print('loss coefficient: ', args.loss_alpha)
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    writer_name = args.test_type+'_cal_log.txt'
    writer = os.path.join(args.results_dir, writer_name)
    with open(writer, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('====== Calibration scores (%s) - %s  ====== \n' % (now, args.data_type))

    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_timebin}
    model_best = MCAT_Surv(**model_dict)

    model_best.load_state_dict(torch.load(args.best_weight_path, weights_only=False))
    model_best.to(device)

    print('Done!')

    results = []

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter=None, category='overall'))

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'is_female': 0.0}, category='male'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'is_female': 1.0}, category='female'))

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'age': (0, 40)}, category='age<40'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'age': (40, 60)}, category='40<=age<60'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'age': (60, 100)}, category='age>60'))

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'mag_level': 1}, category='mag_level1'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'mag_level': 2}, category='mag_level2'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'mag_level': 3}, category='mag_level3'))

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'censorship': 1}, category='alive'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'censorship': 0}, category='dead'))

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'grade': 1}, category='grade1'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'grade': 2}, category='grade2'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'grade': 3}, category='grade3'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'grade': 4}, category='grade4'))

    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'time_bin': 1}, category='risk1'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'time_bin': 2}, category='risk2'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'time_bin': 3}, category='risk3'))
    results.append(calibrate_now(args, datasets, model_best, writer, data_filter={'time_bin': 4}, category='risk4'))

    df_results = pd.DataFrame(results)
    xl_name = args.data_type+'_cal_result.xlsx'
    save_path = os.path.join(args.results_dir, xl_name)
    df_results.to_excel(save_path, index=False)
    






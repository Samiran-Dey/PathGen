from argparse import Namespace
import os
import time
import numpy as np
import torch
import torch.optim as optim

from models.model_coattn import PathoGen
from models.diffusion import GaussianDiffusion
from utils.utils import *
from utils.diffusion_utils import *
from utils.data_utils import *

def get_beta_schedule(num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02, schedule='linear'):
    if schedule == 'linear':
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule == 'cosine':
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(schedule)


def train(datasets: tuple, args: Namespace):
    """   
        train for a single fold
    """
    cur = 1
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    print('\nTraining Fold {}!'.format(cur))

    writer = os.path.join(args.results_dir, 'loss_log.txt')
    with open(writer, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('====== Training Loss (%s)  ======\n' % now)
    
    print('\nInit Model...', end=' ')
    betas = get_beta_schedule(args.T, schedule='linear')
    gen_model = PathoGen(max_time = args.T)
    diff_model = GaussianDiffusion(betas=betas, model=gen_model)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = optim.Adam(gen_model.parameters(), lr=args.lr)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = create_dataset(datasets[0], batch_size=args.batch_size)
    val_loader = create_dataset(datasets[1], batch_size=args.batch_size)
    print('Done!')

    if args.start_epoch != 0:
        print('\n Initializing weights .....', end =' ')
        wt = torch.load(args.weight_path)
        diff_model.model.load_state_dict(wt)
        print('Done!')

    for epoch in range(args.start_epoch, args.max_epochs):
        train_loop(args, cur, epoch, diff_model, train_loader, optimizer, writer)


def test(datasets: tuple, args: Namespace):
    
    epoch = 49
    print('\nepoch: ', epoch)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    writer = os.path.join(args.results_dir, 'test_log_cal.txt')
    with open(writer, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('\n\n\n====== Test result clamp [0,1] %d (%s)  ======\n' % (epoch,now))
    
    print('\nInit Model...', end=' ')
    betas = get_beta_schedule(args.T, schedule='linear')
    gen_model = PathoGen(max_time = args.T)
    
    diff_model = GaussianDiffusion(betas=betas, model=gen_model)
    print('Done!')

    
    print('\nInit Loaders...', end=' ')
    test_loader = create_dataset(datasets[0], batch_size=args.batch_size)
    print('Done!')

    #test_loop(args, epoch, diff_model, test_loader, writer)
    save_prediction(args, epoch, diff_model, test_loader)
    




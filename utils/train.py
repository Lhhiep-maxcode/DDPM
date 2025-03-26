import torch
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from scheduler.linear_scheduler import LinearNoiseScheduler
from data.stanfordcar import StanfordCarDataset
from torch.utils.data import DataLoader
from model.unet import Unet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if args.batch_size:
        config['train_params']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['train_params']['num_epochs'] = args.num_epochs
    if args.lr:
        config['train_params']['lr'] = args.lr
    if args.num_samples:
        config['train_params']['num_samples'] = args.num_samples
    if args.num_grid_rows:
        config['train_params']['num_grid_rows'] = args.num_grid_rows
    if args.ckpt_name:
        config['train_params']['ckpt_name'] = args.ckpt_name
    
    print(config)
    ########################################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'],
                                     timesteps=diffusion_config['num_timesteps'])
    
    dataset = StanfordCarDataset(root_dir=dataset_config['root'],
                                 train_dir=dataset_config['train'],
                                 test_dir=dataset_config['test'])
    dataloader = DataLoader(dataset=dataset, batch_size=train_config['batch_size'], 
                            shuffle=True, num_workers=4)
    
    model = Unet(model_config=model_config).to(device=device)
    model.train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument("--config", dest='config_path', type=str, default="config/config.yaml", 
                        help="Path to config file (default: config/config.yaml)")
    parser.add_argument("--task_name", dest='task_name', type=str, help="Name of the task")
    parser.add_argument("--batch_size", dest='batch_size', type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, help="Number of training epochs")
    parser.add_argument("--lr", dest='lr', type=float, help="Learning rate for optimizer")
    parser.add_argument("--num_samples", dest='num_samples', type=int)
    parser.add_argument("--num_grid_rows", dest='num_grid_rows', type=int)
    parser.add_argument("--ckpt_name", dest="ckpt_name", help="Name of checkpoint file", type=str)
    args = parser.parse_args()
    train(args)
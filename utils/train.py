import torch
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from scheduler.linear_scheduler import LinearNoiseScheduler
from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from model.unet import Unet
from torch.optim import Adam
import torchvision.transforms as transforms
from utils.override_args import override_config


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    config = override_config(config, args, True)
    print(config)
    ########################################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'],
                                     timesteps=diffusion_config['num_timesteps'])
    
    # define image transformation
    transform = transforms.Compose([
        transforms.Resize((model_config['im_size'], model_config['im_size'])),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor (C, H, W), scales [0, 255] → [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize [-1, 1]
    ])

    dataset = CustomDataset(root_path=dataset_config['root'],
                            train_dir=dataset_config['train'],
                            test_dir=dataset_config['test'],
                            transform=transform)
    
    dataloader = DataLoader(dataset=dataset, batch_size=train_config['batch_size'], 
                            shuffle=True, num_workers=4)
    
    model = Unet(model_config=model_config).to(device=device)
    model.train()

    # make a directory to save your model
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # load a checkpoint model if existing
    if os.path.exists(train_config['load_ckpt_path']):
        print("Loading checkpoint model")
        model.load_state_dict(torch.load(train_config['load_ckpt_path'], weights_only=True))

    # define hyperparams and optimizer for training
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # training loop
    for i in range(num_epochs):
        losses = []
        train_pbar = tqdm(dataloader)
        for images in train_pbar:
            optimizer.zero_grad()
            images = images.to(device)

            noise = torch.randn_like(images)
            t = torch.randint(0, diffusion_config['num_timesteps'], (images.shape[0],)).to(device)

            noisy_images = scheduler.forward(images, noise, t)
            noise_pred = model(noisy_images, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Set postfix training loss for tqdm
            train_pbar.set_postfix({"Train loss": loss.item()})
        
        print('Finished epoch:{} | Loss : {:.4f}'.format(i + 1, np.mean(losses)))
        if (i + 1) % train_config['save_every'] == 0:
            print(f"Saving model to {os.path.join(train_config['task_name'], train_config['saved_ckpt_name'])}")
            torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['saved_ckpt_name']))

    print("Done training.....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm training")
    parser.add_argument("--config", dest='config_path', type=str, default="config/config.yaml", 
                        help="Path to config file (default: config/config.yaml)")
    # params for dataset
    parser.add_argument("--root_dir", dest='root_dir', type=str, help="Root directory of dataset")
    parser.add_argument("--train_dir", dest='train_dir', type=str, help="Training directory of dataset")
    parser.add_argument("--test_dir", dest='test_dir', type=str, help="Testing directory of dataset")
    parser.add_argument("--val_dir", dest='val_dir', type=str, help="Validation directory of dataset")
    # params for diffusion
    parser.add_argument("--num_timesteps", dest='num_timesteps', type=int, help="Number of timesteps for diffusion")
    parser.add_argument("--beta_start", dest='beta_start', type=float, help="Beta start for diffusion")
    parser.add_argument("--beta_end", dest='beta_end', type=float, help="Beta end for diffusion")
    # params for model
    parser.add_argument("--im_channels", dest='im_channels', type=int, help="Number of image channels")
    parser.add_argument("--im_size", dest='im_size', type=int, help="Size of image")
    parser.add_argument("--time_emb_dim", dest='time_emb_dim', type=int, help="Time embedding dimension")
    parser.add_argument("--num_down_layers", dest='num_down_layers', type=int, help="Number of down layers in model")
    parser.add_argument("--num_up_layers", dest='num_up_layers', type=int, help="Number of up layers in model")
    parser.add_argument("--num_mid_layers", dest='num_mid_layers', type=int, help="Number of middle layers in model")
    parser.add_argument("--num_heads", dest='num_heads', type=int, help="Number of attention heads")
    parser.add_argument("--dropout", dest='dropout', type=float, help="Dropout rate for model")
    # params for training
    parser.add_argument("--task_name", dest='task_name', type=str, help="Name of the task")
    parser.add_argument("--batch_size", dest='batch_size', type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, help="Number of training epochs")
    parser.add_argument("--lr", dest='lr', type=float, help="Learning rate for optimizer")
    parser.add_argument("--saved_ckpt_name", dest="saved_ckpt_name", help="Name of checkpoint file to be saved", type=str)
    parser.add_argument("--save_every", dest="save_every", help="Save checkpoint every n epochs", type=int)
    parser.add_argument("--load_ckpt_path", dest="load_ckpt_path", help="Path to checkpoint file to be loaded", type=str)

    args = parser.parse_args()
    train(args)
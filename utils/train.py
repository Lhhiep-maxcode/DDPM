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
from torch.optim import Adam
import torchvision.transforms as transforms



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
    if args.task_name:
        config['train_params']['task_name'] = args.task_name
    if args.saved_ckpt_name:
        config['train_params']['saved_ckpt_name'] = args.saved_ckpt_name
    if args.load_ckpt_path:
        config['train_params']['load_ckpt_path'] = args.load_ckpt_path
    if args.save_every:
        config['train_params']['save_every'] = args.save_every
    
    print(config)
    ########################################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'],
                                     timesteps=diffusion_config['num_timesteps'])
    
    transform = transforms.Compose([
        transforms.Resize((model_config['im_size'], model_config['im_size'])),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor (C, H, W), scales [0, 255] → [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize [-1, 1]
    ])

    dataset = StanfordCarDataset(root_dir=dataset_config['root'],
                                 train_dir=dataset_config['train'],
                                 test_dir=dataset_config['test'],
                                 transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=train_config['batch_size'], 
                            shuffle=True, num_workers=4)
    
    model = Unet(model_config=model_config).to(device=device)
    model.train()

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    if os.path.exists(train_config['load_ckpt_path']):
        print("Loading checkpoint model")
        model.load_state_dict(torch.load(train_config['load_ckpt_path']))


    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    for i in range(num_epochs):
        losses = []
        for images in tqdm(dataloader):
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
        
        print('Finished epoch:{} | Loss : {:.4f}'.format(i + 1, np.mean(losses)))
        if (i + 1) % train_config['save_every'] == 0:
            print(f"Saving model to {os.path.join(train_config['task_name'], train_config['saved_ckpt_name'])}")
            torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['saved_ckpt_name']))

    print("Done training.....")


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
    parser.add_argument("--saved_ckpt_name", dest="saved_ckpt_name", help="Name of checkpoint file to be saved", type=str)
    parser.add_argument("--save_every", dest="save_every", help="Save checkpoint every n epochs", type=int)
    parser.add_argument("--load_ckpt_path", dest="load_ckpt_path", help="Path to checkpoint file to be loaded", type=str)
    args = parser.parse_args()
    train(args)
import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from model.unet import Unet
from scheduler.linear_scheduler import LinearNoiseScheduler
from utils.override_args import override_config_infer
from utils.make_vid_from_images import make_vid_from_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, diffusion_config, model_config, infer_config):
    """
    Sample stepwise by going backward one timestep at a time.
    """
    xt = torch.randn((infer_config['num_samples'], model_config['im_channels'], 
                         model_config['im_size'], model_config['im_size'])).to(device)
    model = model.to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), total=(diffusion_config['num_timesteps'])):
        # get model prediction
        timestep = torch.as_tensor(i).unsqueeze(0).expand(infer_config['num_samples']).to(device)
        pred_noise = model(xt, timestep)
        # reverse to timestep t-1
        xt, _ = scheduler.reverse(xt, pred_noise, timestep)

        # denormalize
        mean = torch.tensor([0.5]).view(1, 1, 1).to(device)
        std = torch.tensor([0.5]).view(1, 1, 1).to(device)
        images = xt * std + mean
        images = torch.clamp(images, 0, 1).cpu()
        # save result
        grid = make_grid(images, nrow=infer_config['num_grid_rows'])
        result = torchvision.transforms.ToPILImage()(grid)
        sample_dir = os.path.join(infer_config['task_name'], 'samples')
        os.makedirs(sample_dir, exist_ok=True)  # Creates the directory if it doesn't exist
        result.save(os.path.join(infer_config['task_name'], 'samples', '{}.png'.format(i)))
        result.close()
    
    # create video
    make_vid_from_images(sample_dir, os.path.join(infer_config['task_name'], infer_config['video_name']))

def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = override_config_infer(config, args)
    print(config)
    ###############################################

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    infer_config = config['infer_params']

    # load model with checkpoint
    model = Unet(model_config=model_config).to(device=device)
    model.load_state_dict(torch.load(infer_config['load_ckpt_path'], map_location=device, weights_only=True))

    model.eval()

    # define a scheduler for reverse process
    scheduler = LinearNoiseScheduler(beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'],
                                     timesteps=diffusion_config['num_timesteps'])
    
    with torch.no_grad():
        sample(model, scheduler, diffusion_config, model_config, infer_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path', default='config/config_infer.yaml', type=str, 
                        help="Path to config file (default: config/config_train.yaml)")
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
    # params for inference
    parser.add_argument("--task_name", dest='task_name', type=str, help="Name of the task")
    parser.add_argument("--num_samples", dest='num_samples', type=int, help="Number of samples to generate")
    parser.add_argument("--load_ckpt_path", dest='load_ckpt_path', type=str, help="Path to load checkpoint")
    parser.add_argument("--num_grid_rows", dest='num_grid_rows', type=int, help="Number of grid rows for samples")
    parser.add_argument("--video_name", dest='video_name', type=str, help="Name of the video file")
    args = parser.parse_args()
    infer(args)
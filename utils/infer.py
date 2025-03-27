import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from model.unet import Unet
from scheduler.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, diffusion_config, model_config, infer_config):
    """
    Sample stepwise by going backward one timestep at a time.
    """
    xt = torch.randn((infer_config['num_samples'], model_config['im_channels'], 
                         model_config['im_size'], model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # get model prediction
        timestep = torch.as_tensor(i).unsqueeze(0).expand(infer_config['num_samples']).to(device)
        pred_noise = model(xt, timestep)
        # reverse to timestep t-1
        xt = scheduler.reverse(xt, pred_noise, timestep)

        # denormalize
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        images = xt * std + mean
        images = torch.clamp(images, 0, 1).cpu()
        # save result
        grid = make_grid(images, nrow=infer_config['num_grid_rows'])
        result = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(infer_config['task_name'], 'samples')):
            os.mkdir(os.path.join(infer_config['task_name'], 'samples'))
        result.save(os.path.join(infer_config['task_name'], 'samples', 'x{}.png'.format(i)))
        result.close()



def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

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
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
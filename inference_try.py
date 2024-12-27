import os
import numpy as np
from PIL import Image
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # access image and extract feature
from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.utils import export_to_video

from transformers import BeitFeatureExtractor, BeitForMaskedImageModeling

from src_try.utils import *
from src_try.pipeline import SGI2VPipe
from src_try.model import MyDiT

import argparse

# N: bounding box count, F: frame count
def read_condition(input_dir, config):
    image_path = os.path.join(input_dir, "img.png")
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    image = image.resize((config.width, config.height))     # scale it to specified size
    
    """ traj.shape: (N, 2+F, 2), second dimension is frame
        the first two frames are used to define the bounding boxes, while the following F frames are used to represent the central trajectory 
        third dimension are x and y value """
    traj = np.load(os.path.join(input_dir, "traj.npy")).astype(np.float32)      # load the trajectory data
    traj[:, :, 0] = traj[:, :, 0] * config.width // original_width      # adjust the x-axis scale
    traj[:, :, 1] = traj[:, :, 1] * config.height // original_height    # adjust the y-axis scale
    
    bounding_box = traj[:, :2].reshape(-1, 4)   # N x 4, -1: automate calculate its size of this dimension
    center_traj = traj[:, 2:]   # N x F x 2

    """ convert center trajectory to per_frame top-left/bottom-right coordinates """
    trajectory_points = []
    for j, trajectory in enumerate(center_traj):
        box_traj = []   # store position of this bounding box in every frame
        for i in range(config.num_frames):
            d = center_traj[j][i] - center_traj[j][0]  # d is the offset between two frame
            dx, dy = d[0], d[1]
            
            # bounding_box[j]'s shape: [x1, y1, x2, y2], and top-left coordinates are (x1, y1), bottom-right coordinates are (x2, y2)
            box_traj.append(np.array([bounding_box[j][1] + dy, bounding_box[j][0] + dx,
                                    bounding_box[j][3] + dy, bounding_box[j][2] + dx], dtype = np.float32))
        trajectory_points.append(box_traj)
    return image, trajectory_points

def run(pipe, config, image, trajectory_points):
    """ configure parameters for the dit model """
    pipe.dit.num_inference_steps = config.num_inference_steps   # the steps of denoising progress 
    pipe.dit.heatmap_sigma = config.heatmap_sigma
    pipe.dit.latent_fft_post_merge = config.latent_fft_post_merge
    pipe.dit.latent_fft_ratio = config.fft_ratio
    pipe.dit.optimize_latent_iter = config.optimize_latent_iter
    pipe.dit.optimize_latent_lr = config.optimize_latent_lr
    pipe.dit.optimize_latent_time = config.optimize_latent_time
    pipe.dit.record_layer_sublayer = config.record_layer_sublayer
    
    height, width = config.height, config.width
    motion_bucket_id = 127
    fps = 7
    num_frames = config.num_frames
    seed = config.seed
    generator = torch.manual_seed(seed)
    
    """ execute video generation, pipe is initialize to SGI2VPipe in main function """
    frames = pipe(image, trajectory_points, height, width, num_frames,  
                decode_chunk_size=8, generator=generator, fps=fps, motion_bucket_id=motion_bucket_id,
                noise_aug_strength=0.02).frames[0]
    return frames
    
def main(config, input_dir, output_dir):
    # path check
    assert(os.path.exists(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir_name = os.path.basename(input_dir.rstrip("/\\"))     # remove all "/" and "\" from the end of the path, and extract the dialogue name
    
    print("Reading input condition..")
    image, trajectory_points = read_condition(input_dir, config)
    
    # visualize
    visualize_control(image, trajectory_points=trajectory_points, save_path=os.path.join(output_dir, f"condition_vis_{input_dir_name}.png"))
    
    # load pre_trained image_to_video diffusion models 
    print("Loading Stable Video Diffusion..")
    svd_dir = "stabilityai/stable-video-diffusion-img2vid"
    cache_dir = "./../"
    
    feature_extractor = CLIPImageProcessor.from_pretrained(
        svd_dir, subfolder="feature_extractor", cache_dir=cache_dir,
        torch_dtype=torch.float16, variant="fp16"
    )
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        svd_dir, subfolder="vae", cache_dir=cache_dir,
        torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    requires_grad(vae, False)
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        svd_dir, subfolder="image_encoder", cache_dir=cache_dir,
        torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    requires_grad(image_encoder, False)

    dit_model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-base")
    requires_grad(dit_model, False)
    
    scheduler = EulerDiscreteScheduler.from_pretrained(
        svd_dir, subfolder="scheduler", cache_dir=cache_dir,
        torch_dtype=torch.float16, variant="fp16"
    )
    
    print("Stable Video Diffusion loaded!")
        
    # directly use DiT model to  pretend unet in pipe
    unet = MyDiT(
        image_size=64,
        patch_size=8,
        in_channels=4,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_cross_attention=True
    ).to("cuda")

    # initialize pos_embed
    nn.init.trunc_normal_(unet.pos_embed, std=0.02)
    
    pipe = SGI2VPipe(vae=vae, unet=unet, image_encoder=image_encoder, 
                    scheduler=scheduler, feature_extractor=feature_extractor).to(device="cuda") 
    
    frames = run(pipe, config, image, trajectory_points)    # generate video
    
    export_to_video(frames, os.path.join(output_dir, f"./result_{input_dir_name}.mp4"), fps=7)
    print(f"video is stored to {output_dir}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()
    
    # is same with original paper
    class Config:
        seed = 817
        height, width = 576, 1024
        num_frames = 7
        num_inference_steps = 50
        optimize_latent_time = list(range(30, 46))
        optimize_latent_iter = 5
        optimize_latent_lr = 0.21
        record_layer_sublayer = [(2, 1), (2, 2)]
        heatmap_sigma = 0.4
        fft_ratio = 0.5
        latent_fft_post_merge = True
        
    main(config = Config(), input_dir = args.input_dir, output_dir = args.output_dir)
        
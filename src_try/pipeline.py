import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims

from src_try.model import MyDiT
from src_try.utils import tensor2vid

class SGI2VPipe(StableVideoDiffusionPipeline):
    heatmap = {}
    
    """ *args, **kwargs: provide flexibility, allowing function to accept any number of arguments
        args is a tuple containing all the passed positional arguments, and kwargs is a dictionary containing all the passed keyword arguments """
    def __init__(self, vae, image_encoder, unet, scheduler, feature_extractor):   
        super().__init__(
            vae=vae, 
            image_encoder=image_encoder, 
            unet=unet, 
            scheduler=scheduler, 
            feature_extractor=feature_extractor,
        )    
        self.dit = unet
        
        # self.image_encoder = image_encoder
        self.dit.heatmap_sigma = 0.5
        self.dit.num_inference_steps = 50   # the timesteps of denoising
        self.vae_scale_factor = 8
        
        nn.init.trunc_normal_(self.dit.pos_embed, std=0.02)     # use truncated (截斷) normal distribution to avoid extreme values

        
    # heatmap: represents the importance or weight distribution of certain features
    def get_gaussian_heatmap(self, h, w):   
        if (h, w) in self.heatmap:
            isotropicGrayScaleImage = self.heatmap[(h, w)]  # directly retrieve from the cache
        else:
            sigx = self.dit.heatmap_sigma * (w / 2)     # generate heatmap
            sigy = self.dit.heatmap_sigma * (h / 2)
            cx = w / 2
            cy = h / 2
            isotropicGrayScaleImage = np.zeros((h, w), np.float32)
            for y in range(h):
                for x in range(w):
                    isotropicGrayScaleImage[y, x] = 1 / (2 * np.pi * sigx * sigy) * np.exp(
                        -0.5 * ((x + 0.5 - cx) ** 2 / (sigx ** 2) + (y + 0.5 - cy) ** 2 / (sigy ** 2))
                    )
            isotropicGrayScaleImage = (isotropicGrayScaleImage / np.max(isotropicGrayScaleImage)).astype(np.float32)
            self.heatmap[(h, w)] = isotropicGrayScaleImage
        
        return torch.from_numpy(isotropicGrayScaleImage).cuda()     # convert it to torch tensor and move on to GPU
    
    """ Return the output video frames """
    def __call__(self, image, trajectory_points, height, width, num_frames, min_guidance_scale=1.0, max_guidance_scale=3.0,
                fps=7, generator=None, motion_bucket_id=127, noise_aug_strength=0.02, decode_chunk_size=8):
        """ h, w: height and width of the generated video
            fps: frames per second
            chunk: similar to block
            motion_bucket_id: the motion pattern with ID will be used during the generation process 
            noise_aug_strength: control the intensity of random noise added to the image """
        guidance_scale = max_guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0
        
        batch_size = 1

        
        """ encode image to embedding """
        image_embeddings = self._encode_image(image, "cuda", 1, do_classifier_free_guidance)   # do_classifier_free_guidance: classifier-free guidance enable or not
        
        try:    # preprocess the image, if fail, use video_processor
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")   
        except:
            self.image_processor = self.video_processor
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
            
        """ add noise to the image """
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype).to("cuda")
        image = image + noise_aug_strength * noise
        
        """ encode the image to latent space """
        needs_upcasting = (self.vae.dtype == torch.float16) and (self.vae.config.force_upcast)  # whether let VAE convert float16 to float32
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)    # encode the image into the latent space using a VAE
            image_latents = self._encode_vae_image(image, device="cuda", num_videos_per_prompt=1,
                                                do_classifier_free_guidance=do_classifier_free_guidance)
            image_latents = image_latents.to(image_embeddings.dtype)
        
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)    # convert the VAE model back to float16 to save memory
            
        # expand the dimension of the latent vector to match the number of frames in the generated video
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        """ get the time id, these id are used to control the timestep """
        added_time_ids = self._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength,
                                                image_embeddings.dtype, batch_size, 1,
                                                do_classifier_free_guidance)
        added_time_ids = added_time_ids.to("cuda")
        
        """ set scheduler timesteps """
        self.scheduler.set_timesteps(self.dit.num_inference_steps, device="cuda")
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.dit.in_channels
        
        shape = (batch_size, num_frames, num_channels_latents // 2, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=image.device, dtype=image_embeddings.dtype).to("cuda")
        latents = latents * self.scheduler.init_noise_sigma     # scale initial noise
        
        self.dit.latent_shape = latents.shape
        
        """ Guidance scale per frame, Guidance scale is used to adjust the balance bewteen conditional and unconditional generation """
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)    # shape: (1, nums_frame)
        guidance_scale = guidance_scale.to("cuda", latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size, 1)    # shape: (batch_size, nums_frame)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        
        """ Free CUDA memory """
        self.vae = self.vae.to("cpu")
        self.image_encoder = self.image_encoder.to("cpu")
        torch.cuda.empty_cache()
        
        """ At each timestep, the latent vectors are updated using the noise predicted by the model, gradually generating the final video frames """
        num_warm_steps = max(len(timesteps) - self.dit.num_inference_steps * self.scheduler.order, 0)
        self.num_timesteps = len(timesteps)
        
        with self.progress_bar(total=self.dit.num_inference_steps) as progress_bar:     # set progress bar to visualize progress
            for i, t in enumerate(timesteps):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2, dim=0)  # shape: [2B, F, c, h, w]
                    image_latents_for_model = image_latents.repeat(2, 1, 1, 1, 1)  # repeat in batch dim
                else:
                    latent_model_input = latents
                    image_latents_for_model = image_latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_latents_for_model], dim=2)
                
                with torch.no_grad():
                    noise_pred = self.dit(sample=latent_model_input, timestep=t, encoder_hidden_states=image_embeddings,
                                        added_time_ids=added_time_ids, return_dict=False)[0]    # not return dict, only return first output
                
                if self.guidance_scale is not None and self.guidance_scale.shape[0] > 1:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)    # devide it into unconditional and conditional portion
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # according to predicted noise to update latent vector
                latents = self.scheduler.step(noise_pred, t, latents, s_churn=0.0).prev_sample  # s_churn: control the randomness of the denoising process
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warm_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        """ Restore CUDA memory """
        self.vae = self.vae.to("cuda")
        self.image_encoder = self.image_encoder.to("cuda")
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)    # convert it back to float16
        
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type="np")     # decode the latents vectors into video frames
        self.maybe_free_model_hooks()
        return StableVideoDiffusionPipelineOutput(frames=frames)
        
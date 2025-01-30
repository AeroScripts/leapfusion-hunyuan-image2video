import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from PIL import Image
import cv2

import logging

from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache
from hunyuan_model.vae import load_vae
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def encode_and_save_batch(vae: AutoencoderKLCausal3D, image_file, resolution):
    print(resolution)
    batch =  np.array(Image.open(image_file).convert("RGB"))

    contents = torch.from_numpy(batch).unsqueeze(0)
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.squeeze(0).squeeze(0).cpu().numpy()
    contents = cv2.resize(contents, (resolution[1], resolution[0]), cv2.INTER_LANCZOS4)
    contents = torch.tensor(contents).unsqueeze(0).unsqueeze(0)

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()

        latent = latent * vae.config.scaling_factor

    torch.save(latent, "target_latent.pt")


def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    assert args.vae is not None, "vae checkpoint is required"

    # Load VAE model: HunyuanVideo VAE model is float16
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    print(f"Loaded VAE: {vae.config}, dtype: {vae.dtype}")

    if args.vae_chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    elif args.vae_tiling:
        vae.enable_spatial_tiling(True)

    encode_and_save_batch(vae, args.image, args.video_size)

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument("--image", type=str, default=None, help="the image file to encode")
    parser.add_argument("--video_size", type=int, nargs=2, default=[544, 960], help="video size")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)

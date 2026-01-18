# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
from distutils.log import error
import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
import torch.nn.functional as F
import math

from basicsr.utils.registry import ARCH_REGISTRY

from PIL import Image
import torchvision.transforms as transforms


def tensor_to_image(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor.cpu().squeeze(0))

# 保存图像
def save_image(tensor, file_path):
    image = tensor_to_image(tensor)
    image.save(file_path)

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def tile_process(net, img, tile_size, tile_pad):
    """
    Process image in tiles to avoid OOM.
    """
    batch, channel, height, width = img.shape
    output = torch.zeros_like(img)

    # Number of tiles
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # Loop
    for y_idx in range(tiles_y):
        for x_idx in range(tiles_x):
            y_start = y_idx * tile_size
            x_start = x_idx * tile_size
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Output target area size
            h_out = y_end - y_start
            w_out = x_end - x_start

            # Input padding context
            # We want [y_start, y_end) from output.
            # We need context around it.
            # Crop from input: [y_start - tile_pad, y_end + tile_pad]
            
            y_s_pad = max(0, y_start - tile_pad)
            y_e_pad = min(height, y_end + tile_pad)
            x_s_pad = max(0, x_start - tile_pad)
            x_e_pad = min(width, x_end + tile_pad)

            # Extract
            input_tile = img[:, :, y_s_pad:y_e_pad, x_s_pad:x_e_pad]

            # Calculate missing context at boundaries
            pad_top = y_s_pad - (y_start - tile_pad) 
            pad_bottom = (y_end + tile_pad) - y_e_pad 
            pad_left = x_s_pad - (x_start - tile_pad)
            pad_right = (x_end + tile_pad) - x_e_pad
            
            # Apply reflection padding to restore the full context window
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                input_tile = F.pad(input_tile, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
                
            # Ensure divisibility by 16 (down_factor)
            h_in, w_in = input_tile.shape[2:]
            mod_pad_h = (16 - h_in % 16) % 16
            mod_pad_w = (16 - w_in % 16) % 16
            
            if mod_pad_h > 0 or mod_pad_w > 0:
                input_tile = F.pad(input_tile, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
                
            # Run inference
            with torch.no_grad():
                try:
                    res = net(input_tile)
                    if isinstance(res, (list, tuple)):
                        res = res[-1]
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM error processing tile at y={y_start}, x={x_start}. Try reducing --tile_size.")
                    raise
                    
            # Crop back
            # 1. Remove mod padding
            if mod_pad_h > 0 or mod_pad_w > 0:
                res = res[:, :, :h_in, :w_in]
                
            # 2. Remove context padding
            res_center = res[:, :, tile_pad:tile_pad+h_out, tile_pad:tile_pad+w_out]
            
            # Place in output
            output[:, :, y_start:y_end, x_start:x_end] = res_center
            
            # Clean up
            del input_tile, res, res_center
            
    return output


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='dataset/Real_world/Dataset/MEF')
    parser.add_argument('--result_path', type=str, default='./results/MSUNet/')
    parser.add_argument('--ckpt', type=str, default='weights/LOLBlur.pth')
    # Add tiling arguments
    parser.add_argument('--tile_size', type=int, default=512, help='Tile size for inference')
    parser.add_argument('--tile_pad', type=int, default=32, help='Tile padding')

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    result_root = f'{args.result_path}/{os.path.basename(args.test_path)}'
    
    # Create result directory if it doesn't exist
    os.makedirs(result_root, exist_ok=True)

    net = ARCH_REGISTRY.get('MUWNet')(n_feat=32, nums_stage=5).to(device)
    

    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path)['params']
    net.load_state_dict(checkpoint, strict=True)
    net.eval()

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    img_paths = sorted(list(scandir(args.test_path, suffix=('jpg', 'png', 'bmp','jpeg','JPEG','PNG','JPG'), recursive=True, full_path=True)))

    for img_path in img_paths:
        img_name = img_path.replace(args.test_path+'/', '')
        print(f'Processing: {img_name}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # prepare data
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
        img_t = img_t.unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # Use tile_process instead of direct inference
            output_t = tile_process(net, img_t, args.tile_size, args.tile_pad)
            
            output = tensor2img(output_t, rgb2bgr=True)

        del output_t
        torch.cuda.empty_cache()

        output = output.astype('uint8')
        # save restored img
        save_restore_path = img_path.replace(args.test_path, result_root)
        # Ensure directory exists for nested paths
        os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
        
        imwrite(output, save_restore_path)

    print(f'\nAll results are saved in {result_root}')

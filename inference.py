# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
from distutils.log import error
import os
from turtle import down
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp
import numpy as np
from tqdm import tqdm
import pyiqa
import csv
from datetime import datetime


def tensor_to_image(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor.cpu().squeeze(0))


def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    return x


def compute_metrics(metrics, img_out, img_gt, device):
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(img_out, img_gt).item()
    return results


def main():
    parser = argparse.ArgumentParser()
    # Inference parameters
    parser.add_argument(
        "--test_path",
        type=str,
        default="/home/data/dupf/ddd/LOLBlur/test/low_blur_noise",
        help="Path to the test images",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/home/data/dupf/ddd/LOLBlur/test/high_sharp_scaled",
        help="Path to the ground truth images",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="weights/net_g_latest.pth",
        help="Path to the model checkpoint",
    )

    # Metrics parameters
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips"],
        help="List of IQA metrics to compute",
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        default=False,
        help="Whether to save output images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output images (if save_output is True)",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="./metrics_log.csv",
        help="Path to CSV file to save metrics (appends if exists)",
    )

    args = parser.parse_args()

    # Set up device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clean up paths
    args.test_path = args.test_path.rstrip("/")
    args.gt_path = args.gt_path.rstrip("/")

    # ------------------ set up MUWNet network -------------------
    down_factor = 8  # check_image_size
    net = ARCH_REGISTRY.get("MUWNet")(n_feat=32, nums_stage=5).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path)["params"]
    net.load_state_dict(checkpoint, strict=True)
    net.eval()

    # 新增：从检查点路径中提取权重名称
    weight_name = os.path.basename(ckpt_path)

    # Initialize selected metrics
    available_metrics = {}
    for metric_name in args.metrics:
        try:
            # Special handling for LPIPS metric
            if metric_name == "lpips":
                m = pyiqa.create_metric("lpips-vgg").to(device).eval()
                available_metrics[metric_name] = m
            else:
                m = pyiqa.create_metric(metric_name).to(device).eval()
                available_metrics[metric_name] = m
        except Exception as e:
            print(f"Warning: Failed to load metric '{metric_name}': {e}")

    metric_scores = {k: [] for k in available_metrics}

    # ----------------- scan all the jpg and png images -----------------
    img_paths = sorted(
        list(
            scandir(
                args.test_path, suffix=("jpg", "png"), recursive=True, full_path=True
            )
        )
    )

    # Create output directory if needed
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)

    # -------------------- start to processing ---------------------
    for img_path in tqdm(img_paths, desc="Processing images"):
        img_name = img_path.replace(args.test_path + "/", "")
        try:
            # Load input image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not read input image: {img_path}")
                continue

            # Prepare data
            img_t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
            # 注释掉归一化，因为现在使用0-1范围
            # normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            img_t = img_t.unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():
                # Check image size
                H, W = img_t.shape[2:]
                img_t = check_image_size(img_t, down_factor)
                output_t = net(img_t)
                output_t = output_t[-1]
                output_t = output_t[:, :, :H, :W]

                # Convert output tensor to numpy for metrics calculation
                # 修改min_max从(-1, 1)到(0, 1)
                output_img = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))

                # Prepare tensors for metrics
                img_out_tensor = (
                    torch.from_numpy(
                        np.transpose(output_img.astype(np.float32) / 255.0, (2, 0, 1))
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                # Save output image if needed
                if args.save_output:
                    output_img = output_img.astype("uint8")
                    save_path = os.path.join(args.output_dir, img_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, output_img)

            # Find corresponding ground truth image
            # Try to match by relative path
            rel_path = img_path.replace(args.test_path + "/", "")
            gt_path = os.path.join(args.gt_path, rel_path)

            if not osp.exists(gt_path):
                print(f"Warning: Ground truth image not found: {gt_path}")
                continue

            # Load ground truth image
            img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            if img_gt is None:
                print(f"Warning: Could not read ground truth image: {gt_path}")
                continue

            # Prepare ground truth tensor
            img_gt_tensor = (
                torch.from_numpy(
                    np.transpose(img_gt.astype(np.float32) / 255.0, (2, 0, 1))
                )
                .unsqueeze(0)
                .to(device)
            )

            # Compute metrics
            with torch.no_grad():
                scores = compute_metrics(
                    available_metrics, img_out_tensor, img_gt_tensor, device
                )
                for name, score in scores.items():
                    metric_scores[name].append(score)

            # Clean up GPU memory
            del output_t, img_out_tensor, img_gt_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    # 新增：计算平均分数并保存到CSV
    mean_scores = {}
    for name, scores in metric_scores.items():
        mean_score = sum(scores) / len(scores) if scores else 0.0
        mean_scores[name] = mean_score
        if name == "lpips":
            print(f"{name.upper():<6}: {mean_score:.4f} (lower is better)")
        else:
            print(f"{name.upper():<6}: {mean_score:.4f} (higher is better)")

    # 新增：保存指标到CSV文件
    save_metrics_to_csv(args.metrics_file, weight_name, mean_scores)

    if args.save_output:
        print(f"\nOutput images saved in: {args.output_dir}")
    print(f"Metrics saved to: {args.metrics_file}")


# 新增：保存指标到CSV文件的函数
def save_metrics_to_csv(file_path, weight_name, metrics):
    """
    Save metrics to CSV file with append mode
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)

    # Prepare data row
    row = {"timestamp": timestamp, "weight_name": weight_name}

    # Add all metrics to the row with specific decimal places
    for metric_name, score in metrics.items():
        if metric_name == "psnr":
            row[metric_name] = round(score, 2)  # PSNR保留2位小数
        elif metric_name == "ssim":
            row[metric_name] = round(score, 4)  # SSIM保留4位小数
        elif metric_name == "lpips":
            row[metric_name] = round(score, 3)  # LPIPS保留3位小数
        else:
            row[metric_name] = score  # 其他指标保持原样

    # Get all column names (existing columns + new metrics)
    fieldnames = ["timestamp", "weight_name"] + list(metrics.keys())

    try:
        # Read existing file to get all columns if file exists
        if file_exists:
            existing_rows = []
            with open(file_path, mode="r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_fieldnames = reader.fieldnames

                # Merge fieldnames to include any new metrics
                for field in existing_fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)

                # Read existing data
                for row_data in reader:
                    existing_rows.append(row_data)

            # Write back all data including new row
            with open(file_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write existing rows
                for existing_row in existing_rows:
                    # Fill in missing columns with empty values
                    for field in fieldnames:
                        if field not in existing_row:
                            existing_row[field] = ""
                    writer.writerow(existing_row)

                # Write new row
                writer.writerow(row)
        else:
            # Create new file with headers and data
            with open(file_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")


if __name__ == "__main__":
    main()

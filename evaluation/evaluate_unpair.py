import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from basicsr.utils import img2tensor, tensor2img, scandir
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import csv
from datetime import datetime

# 引入 pyiqa 相关的包
import pyiqa
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def check_image_size(x, down_factor):
    """确保图像尺寸是下采样因子的整数倍，通过 padding 实现。"""
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    return x


@torch.no_grad()
def inference_images(
    test_path, model, device, down_factor, save_output=False, output_dir=None
):
    """
    对指定路径下的所有图像进行模型推理（逐张处理）。
    返回图像名列表和原始的、未经任何转换的 float32 Tensor 结果列表。
    """
    img_paths = sorted(
        list(scandir(test_path, suffix=("jpg", "png"), recursive=True, full_path=True))
    )
    results = []
    names = []

    # Create output directory if needed
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(img_paths, desc="Running inference"):
        img_name = os.path.relpath(img_path, test_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = (
            img2tensor(img / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        )

        H, W = img_tensor.shape[2:]
        img_tensor_padded = check_image_size(img_tensor, down_factor)

        out_tensor = model(img_tensor_padded)
        out_tensor = out_tensor[-1]
        out_tensor = out_tensor[:, :, :H, :W]
        out_tensor = torch.clamp(out_tensor, 0.0, 1.0)

        # Save output image if needed
        if save_output and output_dir:
            output_img = tensor2img(out_tensor, rgb2bgr=True, min_max=(0, 1))
            output_img = output_img.astype("uint8")
            save_path = os.path.join(output_dir, img_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, output_img)

        results.append(out_tensor.cpu())
        names.append(img_name)

    return names, results


@torch.no_grad()
def compute_nr_metrics(raw_image_tensors, image_names, metric_names, device):
    """
    计算无参考(No-Reference)指标
    """
    nr_metrics = [
        m for m in metric_names if DEFAULT_CONFIGS.get(m, {}).get("metric_mode") == "NR"
    ]
    if not nr_metrics:
        raise ValueError("在列表中没有找到有效的 No-Reference (NR) 指标。")

    print(f"\n>>> Evaluating NR metrics: {nr_metrics}")
    all_results = {}

    for m in nr_metrics:
        print(f"\n>>> Computing {m} ...")
        try:
            metric_fn = pyiqa.create_metric(m, as_loss=False, device=device).eval()
            scores = []

            for raw_tensor in tqdm(raw_image_tensors, desc=f"Metric: {m}"):
                uint8_numpy = raw_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                uint8_numpy = (uint8_numpy * 255.0).round().astype(np.uint8)

                reloaded_tensor = (
                    torch.from_numpy(uint8_numpy).permute(2, 0, 1).float() / 255.0
                )
                reloaded_tensor = reloaded_tensor.unsqueeze(0)

                # 计算指标
                tensor_device = reloaded_tensor.to(device)
                score = metric_fn(tensor_device).squeeze().cpu().item()
                scores.append(score)

            avg, std = np.mean(scores), np.std(scores)
            all_results[m] = (avg, std)
            print(f"→ {m}: avg = {avg:.4f}, std = {std:.4f}")
        except Exception as e:
            print(f"Warning: Failed to compute metric '{m}': {e}")

    return all_results


def save_metrics_to_csv(file_path, weight_name, metrics):
    """
    保存指标到CSV文件（只保存平均值）
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)

    # Prepare data row
    row = {"timestamp": timestamp, "weight_name": weight_name}

    # Add only average metrics to the row with specific decimal places
    for metric_name, (avg, std) in metrics.items():
        row[metric_name] = round(avg, 4)

    # Get all column names (only include metric names, not std)
    fieldnames = ["timestamp", "weight_name"]
    fieldnames.extend(sorted(metrics.keys()))

    try:
        # Read existing file to get all columns if file exists
        if file_exists:
            existing_rows = []
            existing_fieldnames = []

            with open(file_path, mode="r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_fieldnames = reader.fieldnames if reader.fieldnames else []

                # Filter out std columns from existing fieldnames
                filtered_fieldnames = []
                for field in existing_fieldnames:
                    if not field.endswith("_std"):
                        filtered_fieldnames.append(field)

                # Read existing data and filter out std columns
                for row_data in reader:
                    filtered_row = {}
                    for key, value in row_data.items():
                        if not key.endswith("_std"):
                            filtered_row[key] = value
                    existing_rows.append(filtered_row)

            # Merge fieldnames: start with existing filtered fieldnames, then add new ones
            # This preserves the original column order for existing columns
            merged_fieldnames = []

            # First add timestamp and weight_name (always at the beginning)
            merged_fieldnames.append("timestamp")
            merged_fieldnames.append("weight_name")

            # Add existing metric columns (excluding timestamp and weight_name)
            for field in filtered_fieldnames:
                if (
                    field not in ["timestamp", "weight_name"]
                    and field not in merged_fieldnames
                ):
                    merged_fieldnames.append(field)

            # Add new metric columns (sorted)
            for field in sorted(fieldnames):
                if field not in merged_fieldnames:
                    merged_fieldnames.append(field)

            fieldnames = merged_fieldnames

            # Write back all data including new row
            with open(file_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write existing rows (only include columns that existed at that time)
                for existing_row in existing_rows:
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


def main():
    parser = argparse.ArgumentParser()
    # Inference parameters
    parser.add_argument(
        "--test_path",
        type=str,
        default="/home/data/dupf/ddd/LOLBlur/real_blur",
        help="Path to the test images",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="weights/net_g_500000.pth",
        help="Path to the model checkpoint",
    )

    # Metrics parameters
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "liqe",
            "arniqa",
            "tres",
            "nima",
            "clipiqa",
            "maniqa",
            "musiq",
            "dbcnn",
            "nrqm",
            "pi",
            "paq2piq",
            "hyperiqa",
            "cnniqa",
            "ilniqe",
            "niqe",
            "brisque",
        ],
        help="List of NR IQA metrics to compute",
    )

    parser.add_argument(
        "--save_output",
        action="store_true",
        default=True,
        help="Whether to save output images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_unpair",
        help="Directory to save output images (if save_output is True)",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="./unpair_metrics_log.csv",
        help="Path to CSV file to save metrics (appends if exists)",
    )

    args = parser.parse_args()

    # Set up device
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clean up paths
    args.test_path = args.test_path.rstrip("/")

    # ------------------ set up MUWNet network -------------------
    down_factor = 16  # check_image_size
    net = ARCH_REGISTRY.get("MUWNet")(n_feat=40, nums_stage=5).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)["params"]
    net.load_state_dict(checkpoint, strict=True)
    net.eval()

    # 从检查点路径中提取权重名称
    weight_name = os.path.basename(ckpt_path)
    print(">>> Model loaded.")

    # --- 步骤 1: 运行模型推理 ---
    names, raw_outputs = inference_images(
        args.test_path, net, device, down_factor, args.save_output, args.output_dir
    )

    results = compute_nr_metrics(raw_outputs, names, args.metrics, device)

    save_metrics_to_csv(args.metrics_file, weight_name, results)

    if args.save_output:
        print(f"\nOutput images saved in: {args.output_dir}")
    print(f"Metrics saved to: {args.metrics_file}")
    print("\n>>> All evaluation completed.")


if __name__ == "__main__":
    main()

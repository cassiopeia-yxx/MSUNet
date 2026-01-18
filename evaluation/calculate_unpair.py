import os
import argparse
import torch
import numpy as np
import pyiqa
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from tqdm import tqdm


def run_nr_metrics(img_dir, metric_names, device):
    # 1) 准备图像列表
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    img_list = []

    # 递归查找所有子目录中的图像文件
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if fname.lower().endswith(valid_ext):
                # 使用相对路径作为文件名，以便后续处理
                rel_path = os.path.relpath(os.path.join(root, fname), img_dir)
                img_list.append(rel_path)

    img_list = sorted(img_list)
    imgs = []
    for name in img_list:
        path = os.path.join(img_dir, name)
        tensor = imread2tensor(path).unsqueeze(0).to(device)
        imgs.append(tensor)
    print(f">>>> Loaded {len(imgs)} images from {img_dir} onto {device}")

    # 2) 筛选出 NR 指标
    nr_metrics = []
    for name in metric_names:
        mode = DEFAULT_CONFIGS.get(name, {}).get("metric_mode", "")
        if mode == "NR":
            nr_metrics.append(name)
    if not nr_metrics:
        raise ValueError("No no-reference (NR) metrics found in your list.")
    print(f">>>> Evaluating NR metrics: {nr_metrics}")

    # 3) 逐指标计算
    with torch.no_grad():
        for m in nr_metrics:
            print(f">>>> Computing {m} ...")
            metric_fn = pyiqa.create_metric(m, as_loss=False, device=device)
            scores = []
            for img in tqdm(imgs):
                score = metric_fn(img).squeeze().cpu().item()
                scores.append(score)
            avg, std = np.mean(scores), np.std(scores)
            print(f"{m}: avg={avg:.4f}, std={std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute no-reference IQA metrics with pyiqa"
    )
    parser.add_argument(
        "--img_dir",
        "-i",
        default=r"F:\\Python\\LLIE&Deblur\\result\\LOL\\Real_world\\MSUNet\\VV",
        help="Path to the folder containing test images",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=[
            "brisque",
        ],
        help="List of no-reference metric names to evaluate (default: ['niqe','liqe'])",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force using CPU even if CUDA is available",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda"
    )

    metrics = args.metrics

    run_nr_metrics(args.img_dir, metrics, device)

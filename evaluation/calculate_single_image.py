import os
import torch
import pyiqa
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def calculate_single_image(img_path, metric_names, device):
    """
    计算单个图片的无参考图像质量评估指标

    Args:
        img_path: 图片文件路径
        metric_names: 指标名称列表
        device: 计算设备 (cuda/cpu)
    """
    # 1) 加载图片
    tensor = imread2tensor(img_path).unsqueeze(0).to(device)
    print(f">>>> Loaded image: {img_path}")
    print(f">>>> Image shape: {tensor.shape}")

    # 2) 筛选出 NR 指标
    nr_metrics = []
    for name in metric_names:
        mode = DEFAULT_CONFIGS.get(name, {}).get("metric_mode", "")
        if mode == "NR":
            nr_metrics.append(name)
        else:
            print(f"Warning: {name} is not a no-reference (NR) metric, skipping...")

    if not nr_metrics:
        raise ValueError("No no-reference (NR) metrics found in your list.")
    print(f">>>> Evaluating NR metrics: {nr_metrics}")

    # 3) 逐指标计算
    results = {}
    with torch.no_grad():
        for m in nr_metrics:
            print(f">>>> Computing {m} ...")
            metric_fn = pyiqa.create_metric(m, as_loss=False, device=device)
            score = metric_fn(tensor).squeeze().cpu().item()
            results[m] = score
            print(f"{m}: {score:.4f}")

    return results


if __name__ == "__main__":
    # ==================== 用户设置区域 ====================
    # 设置图片路径
    img_path = r"F:\\Python\\LLIE&Deblur\\result\\Real_Blur\\MSUNet\\C0349_0040.png"

    metrics = [
        "clipiqa",
        "maniqa",
        "musiq",
        "hyperiqa",
        "niqe",
    ]

    # 是否使用CPU（True=使用CPU, False=使用GPU）
    use_cpu = False
    # ===================================================

    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

    # 计算指标
    results = calculate_single_image(img_path, metrics, device)

    # 打印汇总结果
    print("\n" + "=" * 50)
    print("Summary Results:")
    print("=" * 50)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    print("=" * 50)

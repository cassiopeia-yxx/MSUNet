"""
pyiqa 权重下载脚本
用于提前下载所有需要的指标权重，避免在评估过程中等待下载
"""

import os

# 设置 Hugging Face 镜像（解决网络不可达问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import argparse
import pyiqa
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def create_test_image_tensor(height=256, width=256):
    """
    创建一个伪图像张量用于测试指标计算
    使用渐变色和噪声生成更真实的测试图像

    Args:
        height: 图像高度
        width: 图像宽度

    Returns:
        torch.Tensor: 形状为 (1, 3, height, width) 的张量，值在 [0, 1] 范围内
    """
    # 创建渐变背景
    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    xx, yy = np.meshgrid(x, y)

    # 创建RGB通道
    r = xx * 0.5 + 0.25
    g = yy * 0.5 + 0.25
    b = (xx + yy) * 0.25 + 0.25

    # 添加一些噪声使图像更真实
    noise = np.random.randn(height, width) * 0.05
    r = np.clip(r + noise, 0, 1)
    g = np.clip(g + noise, 0, 1)
    b = np.clip(b + noise, 0, 1)

    # 组合成RGB图像
    img = np.stack([r, g, b], axis=0)  # (3, H, W)
    img = torch.from_numpy(img).float().unsqueeze(0)  # (1, 3, H, W)

    return img


def download_pyiqa_weights(metric_names, device="cuda", test_with_image=True):
    """
    提前下载指定metrics的权重文件

    Args:
        metric_names: 指标名称列表
        device: 使用的设备 (cuda/cpu)
        test_with_image: 是否使用伪图像测试指标计算
    """
    print("=" * 60)
    print("开始下载 pyiqa 权重文件...")
    print("=" * 60)

    # 过滤出有效的NR指标
    nr_metrics = [
        m for m in metric_names if DEFAULT_CONFIGS.get(m, {}).get("metric_mode") == "NR"
    ]

    if not nr_metrics:
        print("警告: 没有找到有效的 No-Reference (NR) 指标")
        return

    print(f"\n需要下载权重的指标: {nr_metrics}\n")

    success_count = 0
    failed_metrics = {}
    dependency_issues = {}

    for metric_name in nr_metrics:
        print(f"正在下载/加载 {metric_name} 权重...")
        try:
            # 创建metric实例，这会自动触发权重下载
            metric_fn = pyiqa.create_metric(metric_name, as_loss=False, device=device)
            metric_fn.eval()

            # 使用伪图像测试指标计算
            if test_with_image:
                test_tensor = create_test_image_tensor().to(device)
                with torch.no_grad():
                    score = metric_fn(test_tensor)
                    # 确保返回的是标量
                    if isinstance(score, torch.Tensor):
                        score_value = score.squeeze().cpu().item()
                    else:
                        score_value = float(score)
                    print(
                        f"✓ {metric_name} 权重下载/加载成功 (测试分数: {score_value:.4f})"
                    )
            else:
                # 简单的随机张量测试
                test_tensor = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    _ = metric_fn(test_tensor)
                print(f"✓ {metric_name} 权重下载/加载成功")

            success_count += 1

        except ImportError as e:
            # 处理缺少依赖的情况
            error_msg = str(e)
            print(f"✗ {metric_name} 缺少依赖: {e}")
            failed_metrics[metric_name] = error_msg
            dependency_issues[metric_name] = error_msg

        except Exception as e:
            # 处理其他错误
            error_msg = str(e)
            print(f"✗ {metric_name} 下载/加载失败: {e}")
            failed_metrics[metric_name] = error_msg

    print("\n" + "=" * 60)
    print(f"下载完成!")
    print(f"成功: {success_count}/{len(nr_metrics)}")

    if failed_metrics:
        print(f"\n失败的指标:")
        for name, error in failed_metrics.items():
            print(f"  - {name}: {error}")

        if dependency_issues:
            print(f"\n缺少依赖的指标 (需要额外安装):")
            print("  提示: 某些指标需要额外的依赖包，例如:")
            print("    - liqe: 可能需要 clip 相关包")
            print("    - clipiqa: 需要 CLIP 模型")
            print("    - musiq: 需要 timm 等包")
            print("  请根据错误信息安装相应的依赖")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="下载 pyiqa 权重文件")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "clipiqa",
        ],
        help="要下载权重的指标列表",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="使用的设备",
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="使用的CUDA设备ID",
    )

    args = parser.parse_args()

    # 设置CUDA设备
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"使用设备: {device}\n")

    download_pyiqa_weights(args.metrics, device)


if __name__ == "__main__":
    main()

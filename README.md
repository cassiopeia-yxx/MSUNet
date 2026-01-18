# MSUNet: A Multi-Scale Unfolding Network for Joint Low-Light Enhancement and Deblurring



## 📋 Requirements

### Environment
- Python 3.10
- PyTorch 2.2.0
- CUDA >= 11.8

### Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

### Setup
1. Install BasicSR in development mode:
```bash
python basicsr/setup.py develop
```

2. Install custom CUDA kernels for selective scan:
```bash
cd kernels/selective_scan && pip install .
```

## 📂 Dataset Preparation

### Supported Datasets
- **LOL-Blur**: Joint low-light and blur dataset
- **LOL-v1**: Low-light enhancement dataset v1
- **LOL-v2**: Low-light enhancement dataset v2 (real & synthetic)

Update the dataset paths in the corresponding YAML configuration files in `options/`:
- `train_LOLBlur.yml`
- `train_LOLv1.yaml`
- `train_LOLv2_real.yaml`
- `train_LOLv2_synthetic.yaml`

## 🏋️ Training

### LOL-Blur Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train_lolblur.py --opt options/train_LOLBlur.yml
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_lolblur.py --opt options/train_LOLBlur.yml --launcher pytorch
```
### LOL-v1 Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train_llie.py --opt options/train_LOLv1.yaml
```

### LOL-v2 Real Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train_llie.py --opt options/train_LOLv2_real.yaml
```

### LOL-v2 Synthetic Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train_llie.py --opt options/train_LOLv2_synthetic.yaml
```


**Note**: Adjust `CUDA_VISIBLE_DEVICES` to specify GPU devices. Checkpoints and logs will be saved in `./checkpoints` by default.

## 🔍 Inference

### LOL-Blur Inference
```bash
python inference_lol_blur.py
```


### Inference with Metrics
```bash
python inference_and_metrics.py
```

## 📊 Evaluation

### Paired Metrics (PSNR, SSIM)
```bash
python evaluation/calculate_pair.py
```

### Unpaired Metrics (NIQE, etc.)
```bash
python evaluation/calculate_unpair.py
```

## 📁 Project Structure

```
MUWNet/
├── basicsr/
│   ├── archs/              # Network architectures
│   │   └── MUWNet_arch.py  # Main network
│   ├── data/               # Data loaders
│   ├── models/             # Training models
│   ├── losses/             # Loss functions
│   ├── metrics/            # Evaluation metrics
│   └── kernels/            # Custom CUDA kernels
├── evaluation/             # Evaluation scripts
├── options/                # Training configurations
├── train_llie.py           # Training script for LLIE
├── train_lolblur.py        # Training script for LOL-Blur
├── inference_llie.py       # Inference script
└── requirements.txt        # Python dependencies
```



## 📧 Contact

For questions or issues, please open an issue or contact [xiaoyao227192@163.com].

## 🙏 Acknowledgements

This project is built upon:
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [Mamba](https://github.com/state-spaces/mamba)
- [PyTorch Image Quality (pyiqa)](https://github.com/chaofengc/IQA-PyTorch)

## 📜 License

This project is released under the MIT License. See LICENSE for details.

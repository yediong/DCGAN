# 🚀 DCGAN-Face-Generation 
*High-Quality Face Synthesis with Deep Convolutional GANs*

[![License](https://img.shields.io/github/license/yediong/DCGAN-Face-Generation)](https://github.com/yediong/DCGAN-Face-Generation/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/yediong/DCGAN-Face-Generation)

---

## 🌟 Overview
State-of-the-art face generation using **Deep Convolutional Generative Adversarial Networks (DCGAN)** trained on the CelebA dataset containing **202,599 celebrity face images**.

> 💡 *This was a time-constrained freelance project completed in just 1 hour, achieving impressive results under pressure!*

---

## 📈 Training Visualization
Watch the generator network evolve through training iterations:

<div align="center">
  <img src="attachments/1.png" alt="Training Progress 1" width="200">
  <img src="attachments/2.png" alt="Training Progress 2" width="200">
  <img src="attachments/3.png" alt="Training Progress 3" width="200">
</div>

---

## 🧰 Technical Stack
- **Framework**: PyTorch
- **Architecture**: DCGAN
- **Dataset**: CelebA Faces
- **Optimization**: Adam Optimizer
- **Hardware**: CUDA-enabled GPU recommended

---

## 🚀 Quick Start Guide

### 📦 Dataset Setup
1. **Install Dependencies**
```bash
sudo apt update && sudo apt install aria2
```

2. **Download HuggingFace CLI**
```bash
wget https://hf-mirror.com/hfd/hfd.sh
chmod +x hfd.sh
```

3. **Configure Environment**
```bash
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows (PowerShell)
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

4. **Fetch Dataset**
```bash
./hfd.sh nielsr/CelebA-faces --dataset
```

### 🛠️ Project Setup
```bash
# Clone repository
git clone https://github.com/yediong/DCGAN-Face-Generation.git

# Process dataset parquet files
python preprocess_parquet.py

# Start training
python DCGAN.py
```

### 🎨 Generate Faces
```bash
python generate_faces.py
# Generated images saved in ./generate_images/
```

---

## ⚙️ Configuration Options
Tune these hyperparameters in `config.py`:
| Parameter      | Description              | Default |
|----------------|--------------------------|---------|
| `batch_size`   | Training batch size      | 128     |
| `lr`           | Learning rate            | 0.0002  |
| `beta1`        | Adam optimizer momentum  | 0.5     |

> ⚠️ *Due to project constraints, parameter tuning was minimal - plenty of room for improvement!*

---

## 📁 Project Structure
```bash
.
├── DCGAN.py          # Training script
├── generate_faces.py # Inference script
├── preprocess_parquet.py # Data processor
└── attachments/      # Visual assets
```

---

## 📚 Additional Resources
- 📘 [Project Report.pdf](Project%20Report.pdf) (Technical Documentation)
- 📘 [DCGAN Implementation Tutorial](https://blog.csdn.net/t1274171989/article/details/134192698) (Chinese)

---

## 💬 Contribution
While this was a freelance project, contributions are welcome for improvement suggestions or enhancements!

---

## 🧾 License
MIT License - see [LICENSE](LICENSE) for details

---

## 💰 Project ROI
> 🎉 Successfully delivered for ¥110 profit - proof that quick GAN projects can deliver value! 💸

---

*Created with ❤️ by [yediong](https://github.com/yediong)*


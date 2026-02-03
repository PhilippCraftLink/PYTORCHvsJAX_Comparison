# Comparison of Deep Learning Frameworks: PyTorch vs. JAX

This project provides a benchmark comparison between **PyTorch** and **JAX**. It implements two popular architectures (ResNet-50 and Vision Transformer) across both frameworks to evaluate performance, memory consumption, and usability.

## Project Structure

The repository contains four main training scripts:

### 1. Vision Transformer (ViT) Comparison
* `Imagenette_VIT_pytorch_final.py`: PyTorch implementation using the `transformers` library to train a ViT on the Imagenette dataset.
* `Imagenette_VIT_JAX_final.py`: JAX/Flax implementation using `FlaxViTForImageClassification` for the same task.

### 2. ResNet-50 Comparison
* `ResNet50_CIFAR_pytorch2.py`: A native PyTorch implementation of ResNet-50 trained on the CIFAR-10 dataset.
* `ResNet50_CIFAR_JAX2.py`: A JAX implementation using `flax.nnx` and `optax` for training ResNet-50 on CIFAR-10.

## Requirements

To run these scripts, you need Python 3.9+ and the dependencies listed in the `requirements.txt`.

The data records have been removed locally here. If the code is executed locally with the data records, they must be added afterwards. 

```bash
pip install -r requirements.txt
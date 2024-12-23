# Particle Image Velocimetry (PIV) Enhancement Framework

## Abstract

Particle Image Velocimetry (PIV) is a cornerstone technique in experimental fluid dynamics, yet its accuracy and reliability can be severely affected by noise and insufficient resolution in the measured velocity fields. In this work, we propose a deep learning framework that integrates signal processing methods with a hybrid U-Net-LSTM convolutional neural network to enhance data quality and resolution. Our approach applies filtering techniques for noise reduction and leverages learned feature extraction for improved particle segmentation, ultimately boosting measurement fidelity. We evaluate this pipeline using a specialized PIV-UQ dataset containing raw images and high-fidelity reference measurements from a stereoscopic system (PIV-HDR). Results indicate that effective denoising in conjunction with the U-Net-LSTM architecture significantly refines pixel-level velocity estimations.

## Keywords

- Deep Optical Flow Learning
- Image Denoising
- PIV
- Experimental Fluid Mechanics


## Baseline UnLiteFlowNet Repository Link

Visit the [UnLiteFlowNet GitHub repository](https://github.com/amenalahassa/UnLiteFlowNet-PIV).


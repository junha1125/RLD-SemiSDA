# RLD-SemiSDA

Official PyTorch implementation of "Is user feedback always informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data" (ECCV 2024)

<div align="center">
  <img src="resources/NBF-RLD.jpeg" width="800"/>
  <div>&nbsp;</div>

[![arXiv](https://img.shields.io/badge/arXiv-2407.15383-b31b1b)](https://arxiv.org/abs/2407.15383) [![project page](https://img.shields.io/badge/project-page-blue)](https://sites.google.com/view/junha/nbf-rld)
</div>

## Overview

This repository contains the implementation of our semi-supervised domain adaptation method that addresses the challenge of biased user feedback in medical image analysis.

## Implementation

#### Chest X-ray Experiments
The chest X-ray experiments implementation is available in the `Chest-X-ray` directory. Please refer to [Chest-X-ray/README.md](Chest-X-ray/README.md) for detailed instructions on:
- Dataset preparation
- Model training
- Adaptation experiments
- Reproduction of our results

#### Image Classification
Implementation for general image classification tasks will be updated soon. The code can be easily built upon [Microsoft's Semi-supervised Learning repository](https://github.com/microsoft/Semi-supervised-learning).

## Citation

If you find this work interesting and useful, please cite our paper:

```bibtex
@inproceedings{song2024nbfrld,
  title={Is user feedback always informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data},
  author={Junha Song and Tae Soo Kim and Junha Kim and Gunhee Nam and Thijs Kooi and Jaegul Choo},
  booktitle={The European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
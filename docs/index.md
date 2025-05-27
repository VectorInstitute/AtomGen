# AtomGen Documentation

Welcome to the documentation for AtomGen, a toolkit for atomistic machine learning and generative modeling. AtomGen provides researchers and developers with tools to explore, experiment, and innovate in the realm of molecular and materials science using state-of-the-art deep learning techniques.

## Overview

AtomGen offers a comprehensive framework for handling atomistic datasets, training various models, and experimenting with different pre-training and fine-tuning tasks. It streamlines the process of working with diverse molecular and materials datasets, enabling large-scale pre-training and task-specific fine-tuning on atomistic data.

### Key Features

- **Data Handling**: Efficient processing and loading of large-scale atomistic datasets.
- **Model Architectures**: Implementation of advanced models like AtomFormer, designed for molecular representation learning.
- **Pre-training**: Support for various pre-training tasks such as Structure to Energy and Forces (S2EF) prediction.
- **Fine-tuning**: Easy adaptation of pre-trained models to downstream tasks using ATOM3D benchmarks.
- **Scalability**: Designed for distributed training on multiple GPUs.

### Datasets

AtomGen supports a variety of datasets, including:

- **S2EF-15M**: A large-scale dataset aggregated from multiple sources (OC20, OC22, ODAC23, MPtrj, SPICE) for pre-training.
- **ATOM3D Benchmarks**: Task-specific datasets for molecular property prediction, including:
  - SMP (Small Molecule Properties)
  - PPI (Protein-Protein Interfaces)
  - RES (Residue Identity)
  - MSP (Mutation Stability Prediction)
  - LBA (Ligand Binding Affinity)
  - LEP (Ligand Efficacy Prediction)
  - PSR (Protein Structure Ranking)
  - RSR (RNA Structure Ranking)

### Models

The implemented model architectures in AtomGen is:

- **AtomFormer**: A transformer encoder model adapted for atomistic data, leveraging 3D spatial information.
- **[SchNet](https://arxiv.org/abs/1706.08566)**: A continuous-filter convolutional neural network for modeling quantum interactions.
- **[TokenGT](https://github.com/jw9730/tokengt)**: Tokenized graph transformer that treats all nodes and edges as independent tokens.

The pre-trained models are based on AtomFormer, which can be fine-tuned on ATOM3D benchmarks for specific molecular property predictions.

### Tasks

AtomGen facilitates various tasks in molecular machine learning:

- **Structure to Energy & Forces (S2EF)**: Predicting energies and forces for atomistic systems.
- **Masked Atom Modeling (MAM)**: Self-supervised learning by masking and predicting atom properties.
- **Coordinate Denoising**: Improving structural predictions by denoising perturbed coordinates.
- **Downstream Tasks**: Fine-tuning on ATOM3D benchmarks for specific molecular property predictions.

## Getting Started

To get started with AtomGen, check out our [User Guide](user_guide.md) for installation instructions, basic usage examples, and more detailed information on training and inference.

For a deep dive into the API, explore the reference documentation for the [API Reference](api.md).

AtomGen is designed to be user-friendly while providing powerful capabilities for atomistic machine learning. Whether you're conducting research, developing new models, or applying machine learning to molecular systems, AtomGen provides a versatile toolkit to support your work.

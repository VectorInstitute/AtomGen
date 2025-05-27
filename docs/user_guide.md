# AtomGen User Guide

Welcome to the AtomGen User Guide. This document provides comprehensive instructions on how to use all components of the AtomGen library for molecular modeling tasks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Loading](#data-loading)
4. [Pretraining](#pretraining)
5. [Fine-tuning](#fine-tuning)
6. [Inference](#inference)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Installation

The package can be installed using uv:

```bash
uv sync
source .venv/bin/activate
```

## Quick Start

Here's a simple example to get you started with AtomGen using a pretrained model to extract features:

```python
import torch
from transformers import AutoModel

# Load a pretrained model
model = AutoModel.from_pretrained("vector-institute/atomformer-base",
                                   trust_remote_code=True)

# Example input data
input_ids = torch.randint(0, 50, (1, 10))
coords = torch.randn(1, 10, 3)
attention_mask = torch.ones(1, 10)

# Extract features
with torch.no_grad():
    output = model(input_ids, coords=coords, attention_mask=attention_mask)

print(output.shape) # Should be (1, 10, 768) for the base model
```

This example demonstrates how to load the pretrained AtomFormer model and use it to extract features from molecular data.

## Data Loading

AtomGen leverages the HuggingFace `datasets` library for data loading. Here are examples of loading some of the available datasets:

```python
from datasets import load_dataset

# Load the S2EF-15M dataset
s2ef_dataset = load_dataset("vector-institute/s2ef-15m")

# Load ATOM3D SMP dataset
smp_dataset = load_dataset("vector-institute/atom3d-smp")

# Load ATOM3D LBA dataset
lba_dataset = load_dataset("vector-institute/atom3d-lba")
```

Dataset structure:
- S2EF-15M: Contains 'input_ids' (atomic numbers), 'coords' (3D coordinates), 'forces', 'formation_energy', 'total_energy', and 'has_formation_energy' fields.
- ATOM3D datasets: Generally contain 'input_ids', 'coords', and task-specific labels. For example, SMP has 20 regression targets, while LBA has a single binding affinity value.

You can inspect the structure of a dataset using:

```python
print(dataset['train'].features)
```

## Pretraining

To pretrain an AtomFormer model, use the `pretrain_s2ef.py` script. Here's an example of how to use it:

```bash
python pretrain_s2ef.py \
    --seed 42 \
    --project "AtomGen" \
    --name "s2ef_15m_train_base_10epochs" \
    --output_dir "./checkpoint" \
    --dataset_dir "./s2ef_15m" \
    --model_config "atomgen/models/configs/atomformer-base.json" \
    --tokenizer_json "atomgen/data/tokenizer.json" \
    --micro_batch_size 8 \
    --macro_batch_size 128 \
    --num_train_epochs 10 \
    --warmup_ratio 0.001 \
    --lr_scheduler_type "cosine" \
    --weight_decay 1.0e-2 \
    --max_grad_norm 5.0 \
    --learning_rate 3e-4 \
    --gradient_checkpointing
```

This script handles the complexities of pretraining, including data loading, model initialization, and training loop management.

## Fine-tuning

For fine-tuning on ATOM3D tasks, use the `run_atom3d.py` script. Here's an example command:

```bash
python run_atom3d.py \
    --model_name_or_path "vector-institute/atomformer-base" \
    --dataset_name "vector-institute/atom3d-smp" \
    --output_dir "./results" \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
```

Key arguments for `run_atom3d.py`:

- `--model_name_or_path`: Pretrained model to start from
- `--dataset_name`: ATOM3D dataset to use for fine-tuning
- `--output_dir`: Directory to save results
- `--batch_size`: Batch size per GPU/CPU for training
- `--learning_rate`: Initial learning rate
- `--num_train_epochs`: Total number of training epochs

## Inference

To use a trained model for inference, you can load it directly from the HuggingFace Hub or from a local directory:

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load from HuggingFace Hub
model = AutoModelForSequenceClassification.from_pretrained("vector-institute/atomformer-base-smp",
                                                           trust_remote_code=True)

# Or load from a local directory
# model = AutoModelForSequenceClassification.from_pretrained("path/to/your/model/directory",
#                                                            trust_remote_code=True)

# Prepare your input data
input_ids = torch.randint(0, 50, (1, 10))
coords = torch.randn(1, 10, 3)
attention_mask = torch.ones(1, 10)

# Run inference
with torch.no_grad():
    output = model(input_ids, coords=coords, attention_mask=attention_mask)

predictions = output[1]
print(predictions.shape)  # Should be (1, 20) for the SMP task
```

This example assumes the model has been fine-tuned on the SMP task. Adjust the model class and output processing based on the specific task you're working with.


## Advanced Features

### Data Collation

The `DataCollatorForAtomModeling` class handles batching of molecular data. Here's how to use it:

```python
from atomgen.data import DataCollatorForAtomModeling

data_collator = DataCollatorForAtomModeling(
    mam=True,  # Enable Masked Atom Modeling
    coords_perturb=0.1,  # Enable coordinate perturbation
    return_lap_pe=True,  # Return Laplacian Positional Encoding
)
```

### Distributed Training

For multi-GPU training, modify your `run_atom3d.py` command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 run_atom3d.py \
    --model_name_or_path "vector-institute/atomformer-base" \
    --dataset_name "vector-institute/atom3d-smp" \
    --output_dir "./results" \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
```

## Troubleshooting

If you encounter out-of-memory errors, try the following:

1. Reduce batch size in the script arguments
2. Enable gradient checkpointing (add `--gradient_checkpointing` to your command)

For more help, please check our [GitHub Issues](https://github.com/your-repo/atomgen/issues) or open a new issue if you can't find a solution to your problem.
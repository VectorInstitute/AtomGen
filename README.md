![atomgen Logo](https://github.com/VectorInstitute/atomgen/blob/main/docs/source/_static/atomgen_logo_text.png?raw=true)
----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/atomgen/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/atomgen/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/atomgen/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/atomgen/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/atomgen/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/atomgen/actions/workflows/docs_deploy.yml)
<!-- [![codecov](https://codecov.io/gh/VectorInstitute/atomgen/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/atomgen) -->
[![license](https://img.shields.io/github/license/VectorInstitute/cyclops.svg)](https://github.com/VectorInstitute/atomgen/blob/main/LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Datasets](#datasets)
- [Models](#models)
- [Tasks](#tasks)
- [Developing](#🧑🏿‍💻-developing)

## Introduction

AtomGen provides a robust framework for handling atomistic graph datasets focusing on transformer-based implementations. We provide utilities for training various models, experimenting with different pre-training tasks, and pre-trained models.

It streamlines the process of aggregation, standardization, and utilization of datasets from diverse sources, enabling large-scale pre-training and generative modeling on atomistic graphs.


## Installation

The package can be installed using poetry:

```bash
python3 -m poetry install
source $(poetry env info --path)/bin/activate
```

## Datasets

AtomGen facilitates the aggregation and standardization of datasets, including but not limited to:

  - **S2EF Datasets**: Aggregated from multiple sources such as OC20, OC22, ODAC23, MPtrj, and SPICE with structures and energies/forces for pre-training.

  - **Misc. Atomistic Graph Datasets**: Including Molecule3D, Protein Data Bank (PDB), and the Open Quantum Materials Database (OQMD).

Currently, AtomGen has pre-processed datasets for the S2EF pre-training task for OC20 and a mixed dataset of OC20, OC22, ODAC23, MPtrj, and SPICE. They have been uploaded to huggingface hub and can be accessed using the datasets API.

## Models

AtomGen supports a variety of models for training on atomistic graph datasets, including:

  - **[SchNet](https://arxiv.org/abs/1706.08566)**: A continuous-filter convolutional neural network for modeling quantum interactions.
  - **[TokenGT](https://github.com/jw9730/tokengt)**: Tokenized graph transformer that treats all nodes and edges as independent tokens.
  - **AtomFormer**: Custom architecture that leverages gaussian pair-wise positional embeddings and self-attention to model atomistic graphs.

## Tasks

Experimentation with pre-training tasks is facilitated through AtomGen, including:

  - **Structure to Energy & Forces**: Predicting energies and forces for atomistic graphs.

  - **Masked Atom Modeling**: Masking atoms and predicting their properties.

  - **Coordinate Denoising**: Denoising atom coordinates.

These tasks are all facilitated through the `DataCollatorForAtomModeling` class and can be used simultaneously or individually.


## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[poetry](https://python-poetry.org/docs/#installation). Hence, make sure it is
installed and then run:

```bash
python3 -m poetry install
source $(poetry env info --path)/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
python3 -m poetry install --with test
```

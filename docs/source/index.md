---
hide-toc: true
---

```{toctree}
:hidden:

user_guide
reference/api/atomgen.data
reference/api/atomgen.models

```

# AtomGen Documentation

Welcome to the documentation for AtomGen, a toolkit for atomistic graph pre-training and generative modeling. AtomGen empowers researchers and developers with tools to explore, experiment, and innovate in the realm of atomistic graph analysis.

## Overview

AtomGen provides a robust framework for handling atomistic graph datasets, training various models, and experimenting with different pre-training tasks. It streamlines the process of aggregation, standardization, and utilization of datasets from diverse sources, enabling large-scale pre-training and generative modeling on atomistic graphs.


### Datasets

AtomGen facilitates the aggregation and standardization of datasets, including but not limited to:

- **S2EF Datasets**: Aggregated from multiple sources such as OC20, OC22, ODAC23, MPtrj, and SPICE with structures and energies for pre-training.

- **Misc. Atomistic Graph Datasets**: Including Molecule3D, Protein Data Bank (PDB), and the Open Quantum Materials Database (OQMD).

Currently, AtomGen has pre-processed datasets for the S2EF pre-training task for OC20 and a mixed dataset of OC20, OC22, ODAC23, MPtrj, and SPICE.  They have been uploaded to huggingface hub and can be accessed using the datasets API.

### Models

AtomGen supports a variety of models for training on atomistic graph datasets, including:

- AtomFormer
- SchNet
- TokenGT

### Tasks

Experimentation with pre-training tasks is facilitated through AtomGen, including:

- **Structure to Energy & Forces**: Predicting energies and forces for atomistic graphs.

- **Masked Atom Modeling**: Masking atoms and predicting their properties.

- **Coordinate Denoising**: Denoising atom coordinates.

These tasks are all facilitated through the ```DataCollatorForAtomModeling``` class and can be used simultaneously or individually.

![atomgen Logo](https://github.com/VectorInstitute/atomgen/blob/main/docs/source/_static/atomgen_logo_text.png?raw=true)
----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml/badge.svg)](https://github.com/VectorInstitute/aieng-template/actions/workflows/docs_deploy.yml)
[![codecov](https://codecov.io/gh/VectorInstitute/aieng-template/branch/main/graph/badge.svg)](https://codecov.io/gh/VectorInstitute/aieng-template)
[![license](https://img.shields.io/github/license/VectorInstitute/aieng-template.svg)](https://github.com/VectorInstitute/aieng-template/blob/main/LICENSE)

## Introduction

AtomGen is a Python package that generates atomistic structures for molecular simulations. The package is designed to be used with the [ASE](https://wiki.fysik.dtu.dk/ase/) package and provides a simple interface to generate structures of various materials.

## Installation

The package can be installed using poetry:

```bash
python3 -m poetry install
source $(poetry env info --path)/bin/activate
```

## Usage

The package can be used to generate structures of various materials. For example, to generate a diamond structure:

```python
from atomgen import Diamond

diamond = Diamond()
diamond.generate()
```


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

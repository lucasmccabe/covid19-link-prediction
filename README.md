[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

# Geodesic and Neural Features for Link Prediction in the COVID-19 Biomedical Knowledge Graph

Public repository for [SMCDC21](https://smc-datachallenge.ornl.gov/) [Challenge 2](https://smc-datachallenge.ornl.gov/2021-challenge-2/) submission.

> Motivated by Challenge 2 of the 5th Annual Smoky Mountains Computational Sciences Data Challenge, we analyze the COVID-19 biomedical knowledge graph. After computing geodesic statistics for all nodes in the network, we present several machine learning pipelines for automated hypothesis generation.

## Table of Contents
* [Table of Contents](#table-of-contents)
* [Repo Organization](#repo-organization)
* [Setup](#setup)
* [kg_browser](#kg_browser)
* [Novel Relations](#novel-relations)
* [Requirements](#requirements)
* [Contact](#contact)
* [License](#license)

## Repo Organization

This repo is organized as follows:

```bash
covid19-link-prediction
├── classifiers  # saved classification models
├── data
│   ├── betweenness     # pre-computed betweenness data
│   ├── embeddings     # trained DeepWalk embeddings
│   ├── graph     # pickled NetworkX graphs
│   ├── og     # dir for provided challenge data
│   ├── other     # dir for intermediary data/computations
│   ├── shortest_paths     # pre-computed APSP data
│   ├── training     # matrices used for training
│   └── validation     # matrices used for validation
├── dev code     # assorted dev notebooks
├── misc     # for assorted supplemental files (currently only the kg_browser image)
├── paper     # submission pre-print
└── kg_browser.py     # utils for browsing processed data/using models/etc.
```

## Setup

1. Clone this repo
2. Download shortest paths data from [here](https://drive.google.com/drive/folders/1vSXfiw09K3RN7gzhBTSOtHZ8_5K61cXE) (~6 GB) and relocate it to [\smcdc-2021-2\data\shortest_paths](https://github.com/lucasmccabe/covid19-link-prediction/tree/main/data/shortest_paths) (see [Repo Organization](#repo-organization))
3. Download validation data from [here](https://drive.google.com/drive/folders/1pC6Z55535CwffG_KXyywhguWwRzmc07-?usp=sharing) (~2/3 GB) and relocate it to [\smcdc-2021-2\data\validation](https://github.com/lucasmccabe/covid19-link-prediction/tree/main/data/validation) (see [Repo Organization](#repo-organization))
4. Download challenge dataset from [here](https://doi.ccs.ornl.gov/ui/doi/346) and relocate it to [\smcdc-2021-2\data\og](https://github.com/lucasmccabe/covid19-link-prediction/tree/main/data/og) (see [Repo Organization](#repo-organization))
5. cd into the covid19-link-prediction directory and run the following in your shell: ```pip pip install -r requirements.txt```

## kg_browser

We provide `kg_browser`, a convenient utility interface for accessing our processed data and models.

![link prediction example](https://raw.githubusercontent.com/lucasmccabe/covid19-link-prediction/main/misc/link%20prediction%20example.png)

For further examples, see the `kg_browser` demo notebook [here](https://github.com/lucasmccabe/covid19-link-prediction/blob/main/Browser%20Demo.ipynb).


## Novel Relations

Our top 1000 proposed novel relations may be found [here](https://github.com/lucasmccabe/covid19-link-prediction/blob/main/dev%20code/link%20prediction/novel_relations.txt). Here are the top 10:

```bash
+-----------------------+------------------------------+
| Edge                  |  Estimated Link Probability  |
+=======================+==============================+
| C0035236 <-> C1441604 |           0.999488           |
+-----------------------+------------------------------+
| C0027362 <-> C0020967 |           0.999487           |
+-----------------------+------------------------------+
| C0003062 <-> C0012754 |           0.999484           |
+-----------------------+------------------------------+
| C0086418 <-> C0027934 |           0.999484           |
+-----------------------+------------------------------+
| C0006104 <-> C0333230 |           0.999484           |
+-----------------------+------------------------------+
| C1314650 <-> C2700280 |           0.999464           |
+-----------------------+------------------------------+
| C0543467 <-> C0265883 |           0.99945            |
+-----------------------+------------------------------+
| C0582175 <-> C2697883 |           0.999444           |
+-----------------------+------------------------------+
| C1320226 <-> C0401805 |           0.999403           |
+-----------------------+------------------------------+
| C0206031 <-> C0038454 |           0.99938            |
+-----------------------+------------------------------+
```

## Requirements
This project was created with:

- `powerlaw==1.4.6`
- `scikit_network==0.23.1`
- `numpy==1.21.1`
- `networkx==2.5`
- `pandas==1.2.4`
- `seaborn==0.11.0`
- `matplotlib==3.3.2`
- `scipy==1.7.0`
- `joblib==0.17.0`

## Contact
- Lucas Hurley McCabe ([email](mailto:lucasmccabe@gwu.edu))

## License
[MIT](https://choosealicense.com/licenses/mit/)

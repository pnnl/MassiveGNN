# MassiveGNN
## Description
MassiveGNN is a framework designed to accelerate the training of Graph Neural Networks (GNNs) on large-scale, massively connected distributed graphs. Built on top of DistDGL, it utilizes a dynamic prefetching and eviction strategy to optimize remote data fetching and processing and significantly reduces the training time for GNN models on distributed memory systems, enabling efficient and scalable training of large graph datasets.

## Installation

To install the required dependencies and DGL (Deep Graph Library), follow the steps below:

### Prerequisites

- Python 
- Torch with CUDA support
- DGL
- NUMBA

### Installing DGL

Follow the instructions provided on the [DGL official installation page](https://www.dgl.ai/pages/start.html) to install DGL with the appropriate configuration for your environment (CPU/GPU).

### Installing MassiveGNN

1. Clone the MassiveGNN repository:

    ```bash
    git clone https://github.com/yourusername/massivegnn.git
    cd massivegnn
    ```

2. Create and activate the conda environment using the provided `env.yml` file:

    ```bash
    conda env create -f env.yml
    conda activate dgl-dev-gpu-117
    ```

## Running MassiveGNN

1. Partition Graph using DGL
2. To run MassiveGNN on a distributed system, modify the job scripts provided in the repo based on your cluster.

## How to Cite
If you use MassiveGNN in your research, please cite our paper:
```
@inproceedings{yourcitation2024,
  title={Efficient training of GNN for massively connected distributed graphs using prefetching},
  author={Your Name and Collaborators},
  booktitle={Proceedings of the Conference},
  year={2024},
  organization={IEEE/ACM}
}
```
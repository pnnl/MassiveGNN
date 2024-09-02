# MassiveGNN

Efficient training of Graph Neural Networks (GNN) for massively connected distributed graphs using prefetching.

## Description

MassiveGNN is a high-performance implementation designed to accelerate the training of Graph Neural Networks (GNNs) on large-scale, massively connected distributed graphs. By utilizing prefetching techniques, MassiveGNN optimizes data loading and processing, significantly reducing the training time for GNN models on distributed systems.

## Installation

To install the required dependencies and DGL (Deep Graph Library), follow the steps below:

### Prerequisites

- Python 3.7 or higher
- CUDA 10.1 or higher (for GPU support)
- Git

### Installing DGL

Follow the instructions provided on the [DGL official installation page](https://www.dgl.ai/pages/start.html) to install DGL with the appropriate configuration for your environment (CPU/GPU).

### Installing MassiveGNN

1. Clone the MassiveGNN repository:

    ```bash
    git clone https://github.com/yourusername/massivegnn.git
    cd massivegnn
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment:

    ```bash
    source setup_env.sh  # if you have an environment setup script
    ```

## Running MassiveGNN

To run MassiveGNN on a distributed system, follow these steps:

### Running on a Single Node

You can start by running the model on a single node to verify the setup:

```bash
python train.py --config configs/single_node.yaml

# MassiveGNN Docker Run

This guide provides the steps to set up and run MassiveGNN using Docker on an AWS ParallelCluster.

## Getting Started

### 1. Create a ParallelCluster on AWS
- #TODO: 

### 2. Setup Docker on the Head Node
- SSH into the head node of the cluster.
- Install Docker if it is not already installed:
  ```bash
  sudo yum install docker -y
  sudo systemctl start docker
  sudo usermod -aG docker ec2-user
  ```

### 3. Prerequisite
- Docker setup on each compute node.
- Install NVIDIA Container Toolkit. Please refer to the official [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Install AWS CLI.
    ```bash
    sudo yum install aws-cli -y
    aws configure # Provide your AWS access key, secret access key, region, and output format.
    aws s3 ls s3://massivegnn # Test access to the MassiveGNN S3 bucket
    ```
## Run MassiveGNN
To run MassiveGNN, use the `set_params.sh` script to set up and submit the job. To update training hyperparameters, modify `gpu.sh/cpu.sh`.  

```bash
Usage: set_params.sh [MODE] [HIT_RATE] [MODEL] [FP] [DELTA] [ALPHAS] [DATASET_NAME] [NUM_NODES] [NUM_TRAINERS] [NUM_SAMPLER_PROCESSES] [QUEUE] [LOGS_DIR] [PROJ_PATH] [PARTITION_DIR] [PARTITION_METHOD]

Arguments:
  MODE                 Execution mode, either 'cpu' or 'gpu'.
  HIT_RATE             Hit rate flag, 'true' or 'false'.
  MODEL                Model name to be used. Currently accepts 'sage' or 'gat'.
  FP                   % halo nodes to prefetch while initializing buffer (e.g., '0.5').
  DELTA                Eviction interval.
  ALPHAS               Alpha value (e.g., '0.05'). Alpha is calculated as 1-delta.
  DATASET_NAME         Name of the dataset (e.g., 'ogbn-products').
  NUM_NODES            Number of nodes to be used (e.g., '2 4 8').
  NUM_TRAINERS         Number of trainers to be used.
  NUM_SAMPLER_PROCESSES Number of sampler processes to be used.
  QUEUE                SLURM queue name (e.g., 'regular' or 'debug').
  LOGS_DIR             Path to SLURM logs.
  PROJ_PATH            Path to the project directory.
  PARTITION_DIR        Directory where the partitioned graphs are stored.
  AWS                  Path to AWS Credential

Example:
  set_params.sh gpu true sage 0.25 32 0.005 ogbn-arxiv 2 1 0 regular ~/MassiveGNN ~/MassiveGNN ~/MassiveGNN/partitions metis ~/.aws/credentials
```
**Note:** Partitions are available in the `s3://massivegnn` bucket. Additionally, sample partitions of the `ogbn-arxiv` dataset are provided locally in the `~/MassiveGNN/partitions` directory.
 
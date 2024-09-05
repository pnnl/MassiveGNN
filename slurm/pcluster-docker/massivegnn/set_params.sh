#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [MODE] [HIT_RATE] [MODEL] [FP] [DELTA] [ALPHAS] [DATASET_NAME] [NUM_NODES] [NUM_TRAINERS] [NUM_SAMPLER_PROCESSES] [QUEUE] [LOGS_DIR] [PROJ_PATH] [PARTITION_DIR] [PARTITION_METHOD]"
    echo
    echo "Arguments:"
    echo "  MODE                 Execution mode, either 'cpu' or 'gpu'."
    echo "  HIT_RATE             Hit rate flag, 'true' or 'false'."
    echo "  MODEL                Model name to be used. Currently accepts 'sage' or 'gat'."
    echo "  FP                   % halo nodes to prefetch while initializing buffer (e.g., '0.5')."
    echo "  DELTA                Eviction interval."
    echo "  ALPHAS               Alpha value (e.g., '0.05'). Alpha is calculated as 1-delta."
    echo "  DATASET_NAME         Name of the dataset (e.g., 'ogbn-products')."
    echo "  NUM_NODES            Number of nodes to be used (e.g., '2 4 8')."
    echo "  NUM_TRAINERS         Number of trainers to be used."
    echo "  NUM_SAMPLER_PROCESSES Number of sampler processes to be used."
    echo "  QUEUE                SLURM queue name (e.g., 'regular' or 'debug')."
    echo "  LOGS_DIR             Path to SLURM logs."
    echo "  PROJ_PATH            Path to the project directory."
    echo "  PARTITION_DIR        Directory where the partitioned graphs are stored."
    echo "  AWS                  Path to AWS Credential"
    echo
    echo "Example:"
    echo "  $0 gpu true sage 0.25 32 0.005 ogbn-arxiv 2 1 0 regular ~/MassiveGNN ~/MassiveGNN ~/MassiveGNN/partitions metis ~/.aws/credentials"
    echo
}


# Check if help is requested
if [[ $1 == "-h" || $1 == "--help" ]]; then
    show_help
    exit 0
fi

# CMD arguments
MODE=$1
HIT_RATE=$2
MODEL=$3
FP=$4
DELTA=$5
ALPHAS=$6
DATASET_NAME=$7
NUM_NODES=$8
NUM_TRAINERS=$9
NUM_SAMPLER_PROCESSES=${10}
QUEUE=${11}
LOGS_DIR=${12}
PROJ_PATH=${13}
PARTITION_DIR=${14}
PARTITION_METHOD=${15}
AWS_FILE=${16}

echo $AWS_FILE
# Extracting the values from the credentials file
AWS_ACCESS_KEY_ID=$(grep -A 3 "\[622093707179_AdministratorAccess\]" "$AWS_FILE" | grep aws_access_key_id | cut -d '=' -f 2 | xargs)
AWS_SECRET_ACCESS_KEY=$(grep -A 3 "\[622093707179_AdministratorAccess\]" "$AWS_FILE" | grep aws_secret_access_key | cut -d '=' -f 2 | xargs)
AWS_SESSION_TOKEN=$(grep -A 3 "\[622093707179_AdministratorAccess\]" "$AWS_FILE" | grep aws_session_token | cut -d '=' -f 2 | xargs)
PARTITION_DIR="${PARTITION_DIR}/${PARTITION_METHOD}/${DATASET_NAME}/${NUM_NODES}_parts/"

# Validate that all required arguments are provided
if [ -z "$MODE" ] || [ -z "$HIT_RATE" ] || [ -z "$MODEL" ] || [ -z "$FP" ] || [ -z "$DELTA" ] || [ -z "$ALPHAS" ] || [ -z "$DATASET_NAME" ] || [ -z "$NUM_NODES" ] || [ -z "$NUM_TRAINERS" ] || [ -z "$NUM_SAMPLER_PROCESSES" ] || [ -z "$QUEUE" ] || [ -z "$LOGS_DIR" ] || [ -z "$PROJ_PATH" ] || [ -z "$PARTITION_DIR" ]; then
    echo "Error: One or more required arguments are missing."
    show_help
    exit 1
fi


# Loop over each combination of prefetch fraction, delta, and alpha
for fp in "${FP[@]}"; do
    for delta in "${DELTA[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            echo "Submitting job for $NUM_NODES nodes with prefetch fraction $fp, delta $delta, alpha $alpha"
            if [ "$MODE" == "cpu" ]; then
                bash submit.sh cpu gloo "$fp" "$delta" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$NUM_NODES" "$NUM_TRAINERS" "$NUM_SAMPLER_PROCESSES" "$MODEL" "$QUEUE" "$LOGS_DIR" "$PROJ_PATH" "$PARTITION_DIR" \
                "$PARTITION_METHOD" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_SESSION_TOKEN"
            elif [ "$MODE" == "gpu" ]; then
                cmd="submit.sh gpu nccl "$fp" "$delta" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$NUM_NODES" "$NUM_TRAINERS" "$NUM_SAMPLER_PROCESSES" "$MODEL" "$QUEUE" "$LOGS_DIR" "$PROJ_PATH" "$PARTITION_DIR" \
                "$PARTITION_METHOD" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$AWS_SESSION_TOKEN""
                bash $cmd
            else
                echo "Invalid mode: choose either 'cpu' or 'gpu'"
                exit 1
            fi
        done
    done
done

#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [MODE] [HIT_RATE] [MODEL] [FP] [DELTA] [ALPHAS] [DATASET_NAME] [NUM_NODES] [QUEUE] [LOGS_DIR]"
    echo
    echo "Arguments:"
    echo "  MODE           Execution mode, either 'cpu' or 'gpu'."
    echo "  HIT_RATE       Hit rate flag, 'true' or 'false'"
    echo "  MODEL          Model name to be used. Currently accepts 'sage' or 'gat'."
    echo "  FP             % halo nodes to prefetch while initializing buffer (e.g., '0.5')."
    echo "  DELTA          Eviction interval."
    echo "  ALPHAS         Alpha value (e.g., '0.05'). Alpha is calculated as 1-delta."
    echo "  DATASET_NAME   Name of the dataset (e.g., 'ogbn-products')."
    echo "  NUM_NODES      Number of nodes to be used (e.g., '2 4 8')."
    echo "  QUEUE          SLURM queue name (e.g., 'regular' or 'debug')."
    echo "  LOGS_DIR       Path to slurm logs."
    echo
    echo "Example:"
    echo "  $0 gpu true sage 0.25 32 0.005 ogbn-products '2 4 8' regular '~/MassiveGNN'"
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
QUEUE=$9
LOGS_DIR=${10}

# Validate that all required arguments are provided
# Validate that all required arguments are provided
if [ -z "$MODE" ] || [ -z "$HIT_RATE" ] || [ -z "$MODEL" ] || [ -z "$FP" ] || [ -z "$DELTA" ] || [ -z "$ALPHAS" ] || [ -z "$DATASET_NAME" ] || [ -z "$NUM_NODES" ] || [ -z "$QUEUE" ] || [ -z "$LOGS_DIR" ]; then
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
                bash submit.sh cpu gloo "$fp" "$delta" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$NUM_NODES" "$MODEL" "$QUEUE" "$LOGS_DIR"
            elif [ "$MODE" == "gpu" ]; then
                bash submit.sh gpu nccl "$fp" "$delta" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$NUM_NODES" "$MODEL" "$QUEUE" "$LOGS_DIR"
            else
                echo "Invalid mode: choose either 'cpu' or 'gpu'"
                exit 1
            fi
        done
    done
done

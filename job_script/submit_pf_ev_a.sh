#!/bin/bash
pf=(0.25)
periods=(0)
alphas=(0)

# CMD arguments
MODE=$1
HIT_RATE=$2
MODEL=$3
DATASET_NAME="ogbn-products"
NUM_NODES="2"
QUEUE="debug"


if [ -z "$MODE" ]; then
    echo "Please provide mode"
    exit 1
fi

# if hitrate is not provided, throw error
if [ -z "$HIT_RATE" ]; then
    echo "Please provide hit rate flag"
    exit 1
fi

# Loop over each period and alpha combination
for pf in "${pf[@]}"; do
    for period in "${periods[@]}"; do
        for alpha in "${alphas[@]}"; do

            # SKIP if period = 256 and alpha = 0.05
            # if [ "$period" == "256" ] && [ "$alpha" == "0.05" ]; then
            #     continue
            # fi 
            echo "Submitting job for $NODES nodes with prefetch fraction $pf, period $period, alpha $alpha"
            if [ "$MODE" == "cpu" ]; then
                bash submit_dgl_ex.sh cpu gloo "$pf" "$period" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$NUM_NODES" "$MODEL" "$QUEUE"
            elif [ "$MODE" == "gpu" ]; then
                bash submit_dgl_ex.sh gpu nccl "$pf" "$period" "$alpha" "$HIT_RATE" "$DATASET_NAME" "$NUM_NODES" "$MODEL" "$QUEUE"
            else
                echo "Invalid mode: choose either 'cpu' or 'gpu'"
                exit 1
            fi
        done
    done
done
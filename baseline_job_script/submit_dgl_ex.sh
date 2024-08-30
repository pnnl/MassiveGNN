#!/bin/bash
DATASET_NAME="ogbn-products"
NUM_NODES="2"
NUM_TRAINERS="4"
NUM_SAMPLER_PROCESSES="0"
PARTITION_METHOD="metis"

MODE=$1 # pass 'cpu' or 'gpu' as argument
PROFILE=false
MODEL=$3    # pass 'gat' or 'sage' as argument

if [ "$MODE" == "cpu" ]; then
    BACKEND=$2
    # if backend is not passed as argument, throw error
    if [ -z "$BACKEND" ]; then
        echo "Backend not passed as argument"
        exit 1
    fi
    if [ "$4" == "profile" ]; then
        PROFILE=true
    fi
    SCRIPT="dgl_ex_cpu.sh"
    LOGNAME="logs_perlmutter_cpu_${BACKEND}"
    
elif [ "$MODE" == "gpu" ]; then
    BACKEND=$2
    SCRIPT="dgl_ex_gpu.sh"
    if [ -z "$BACKEND" ]; then
        echo "Backend not passed as argument"
        exit 1
    fi
    LOGNAME="logs_perlmutter_gpu_${BACKEND}"
    # accept an optional argument for profile
    if [ "$4" == "profile" ]; then
        PROFILE=true
    fi
else
    echo "Invalid mode: choose either 'cpu' or 'gpu'"
    exit 1
fi

for DATASET in $DATASET_NAME; do
    if [ "$MODEL" == "gat" ]; then 
         LOGS_DIR="/global/cfs/cdirs/m4626/Distributed_DGL/dgl_ex/experiments/logs/${DATASET}/release/distdgl/${LOGNAME}/gat"
    fi
    if [ "$MODEL" == "sage" ]; then
         LOGS_DIR="/global/cfs/cdirs/m4626/Distributed_DGL/dgl_ex/experiments/logs/${DATASET}/release/distdgl/${LOGNAME}"
    fi
    IP_CONFIG_DIR="${LOGS_DIR}/ip_config"

    mkdir -p $LOGS_DIR
    mkdir -p $IP_CONFIG_DIR
    for PARTITION in $PARTITION_METHOD; do
        for NODES in $NUM_NODES; do
            for SAMPLER_PROCESSES in $NUM_SAMPLER_PROCESSES; do
                for TRAINERS in $NUM_TRAINERS; do
                    JOBNAME="${DATASET}_${PARTITION}_n${NODES}_samp${SAMPLER_PROCESSES}_trainer${TRAINERS}"
                    OUTFILE="${LOGS_DIR}/${JOBNAME}_%j.out"
                    ERRFILE="${LOGS_DIR}/${JOBNAME}_%j.err"
                    SUMMARYFILE="${LOGS_DIR}/${JOBNAME}"
                    IP_CONFIG_FILE="${IP_CONFIG_DIR}/ip_config_${JOBNAME}"
                    PROFILE_DIR="/pscratch/sd/s/sark777/Distributed_DGL/profiles/${PARTITION}/${DATASET}/${MODE}/${BACKEND}/${NODES}_parts/samp_${SAMPLER_PROCESSES}/trainer_${TRAINERS}"

                    # create Profile Directory if it doesn't exist
                    mkdir -p $PROFILE_DIR

                    echo "Submitting job $JOBNAME with the following parameters:"
                    echo "Dataset: $DATASET"
                    echo "Partition Method: $PARTITION"
                    echo "Number of Nodes: $NODES"
                    echo "Number of Sampler Processes: $SAMPLER_PROCESSES"
                    echo "Number of Trainers: $TRAINERS"
                    echo "Using $BACKEND backend"
                    echo "Summary file: $SUMMARYFILE"
                    echo "Running script: $SCRIPT"
                    echo "IP Config file: $IP_CONFIG_FILE"
                    echo "Profile Directory: $PROFILE_DIR"
                    echo "Logging to: $OUTFILE"
                    # if mode is gpu
                    if [ "$MODE" == "gpu" ]; then
                        CMD="sbatch -N $NODES --job-name $JOBNAME -o $OUTFILE -e $ERRFILE $SCRIPT $DATASET $PARTITION \
                        $NODES $SAMPLER_PROCESSES $SUMMARYFILE $IP_CONFIG_FILE $TRAINERS $BACKEND $PROFILE $MODEL"
                    elif [ "$MODE" == "cpu" ]; then
                        CMD="sbatch -N $NODES --job-name $JOBNAME -o $OUTFILE -e $ERRFILE $SCRIPT $DATASET $PARTITION \
                        $NODES $SAMPLER_PROCESSES $SUMMARYFILE $IP_CONFIG_FILE $BACKEND $PROFILE_DIR $TRAINERS $MODEL"
                    fi               
                    # Submit the job
                    eval $CMD
                done
            done
        done
    done
done
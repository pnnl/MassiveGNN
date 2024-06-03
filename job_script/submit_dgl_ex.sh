#!/bin/bash

MODE=$1 # pass 'cpu' or 'gpu' as argument
PROFILE=false
EVICTION_PERIOD=$4
PREFETCH_FRACTION=$3
ALPHA=$5
HIT_RATE=$6
DATASET_NAME=$7
NUM_NODES=$8
MODEL=$9    # pass 'gat' or 'sage' as argument
QUEUE=${10}
NUM_TRAINERS="4"
NUM_SAMPLER_PROCESSES="0"
PARTITION_METHOD="metis"

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
    if [ "$5" == "profile" ]; then
        PROFILE=true
    fi
else
    echo "Invalid mode: choose either 'cpu' or 'gpu'"
    exit 1
fi

for DATASET in $DATASET_NAME; do
    if [ "$MODEL" == "gat" ]; then 
         LOGS_DIR="/global/cfs/cdirs/m4626/Distributed_DGL/dgl_ex/experiments/logs/${DATASET}/scratch_built/cache_evict/${LOGNAME}/gat/pf_${PREFETCH_FRACTION}/${EVICTION_PERIOD}_period_${PREFETCH_FRACTION}_fraction_${ALPHA}_alpha"
    fi
    if [ "$MODEL" == "sage" ]; then
         LOGS_DIR="/global/cfs/cdirs/m4626/Distributed_DGL/dgl_ex/experiments/logs/${DATASET}/scratch_built/cache_evict/${LOGNAME}/overlap/pf_${PREFETCH_FRACTION}/${EVICTION_PERIOD}_period_${PREFETCH_FRACTION}_fraction_${ALPHA}_alpha"
    fi
   
    IP_CONFIG_DIR="${LOGS_DIR}/ip_config"
    mkdir -p $LOGS_DIR
    mkdir -p $IP_CONFIG_DIR
    for PARTITION in $PARTITION_METHOD; do
        for NODES in $NUM_NODES; do
            for SAMPLER_PROCESSES in $NUM_SAMPLER_PROCESSES; do
                for TRAINERS in $NUM_TRAINERS; do
                    NAME="${DATASET}_${PARTITION}_n${NODES}_samp${SAMPLER_PROCESSES}_trainer${TRAINERS}"
                    JOBNAME="${MODE}_PF{$PREFETCH_FRACTION}_P{$EVICTION_PERIOD}_a{$ALPHA}_${HIT_RATE}_${MODEL}_${NAME}"
                    OUTFILE="${LOGS_DIR}/${NAME}_%j.out"
                    ERRFILE="${LOGS_DIR}/${NAME}_%j.err"
                    SUMMARYFILE="${LOGS_DIR}/${NAME}"
                    IP_CONFIG_FILE="${IP_CONFIG_DIR}/ip_config_${NAME}"
                    PROFILE_DIR="/pscratch/sd/s/sark777/Distributed_DGL/profiles/${PARTITION}/${DATASET}/${MODE}/${BACKEND}/${NODES}_parts/samp_${SAMPLER_PROCESSES}/trainer_${TRAINERS}"

                    # create Profile Directory if it doesn't exist
                    mkdir -p $PROFILE_DIR

                    if [ "$MODEL" == "gat" ]; then
                        if [ "$QUEUE" == "debug" ]; then
                            TIME="00:30:00"
                        elif [ "$QUEUE" == "regular" ]; then
                            if [ "$DATASET" == "ogbn-papers100M" ]; then
                                if [ "$NODES" == "2" ]; then
                                    TIME="06:00:00"
                                elif [ "$NODES" == "4" ]; then
                                    TIME="03:30:00"
                                elif [ "$NODES" == "8" ]; then
                                    TIME="03:00:00"
                                elif [ "$NODES" == "16" ]; then
                                    TIME="01:30:00"
                                elif [ "$NODES" == "32" ]; then
                                    TIME="00:45:00"
                                elif [ "$NODES" == "64" ]; then
                                    TIME="00:50:00"
                                fi
                            else
                                TIME="00:30:00"
                            fi
                        fi
                    elif [ "$MODEL" == "sage" ]; then
                        if [ "$QUEUE" == "debug" ]; then
                            TIME="00:30:00"
                        elif [ "$QUEUE" == "regular" ]; then
                            if [ "$DATASET" == "ogbn-papers100M" ]; then
                                if [ "$NODES" == "2" ]; then
                                    TIME="03:00:00"
                                elif [ "$NODES" == "4" ]; then
                                    TIME="02:45:00"
                                elif [ "$NODES" == "8" ]; then
                                    TIME="01:00:00"
                                elif [ "$NODES" == "16" ]; then
                                    TIME="00:45:00"
                                elif [ "$NODES" == "32" ]; then
                                    TIME="00:40:00"
                                elif [ "$NODES" == "64" ]; then
                                    TIME="00:40:00"
                                fi
                            else
                                TIME="00:30:00"
                            fi
                        fi
                    fi
                    echo "-----------------------------------------------------"
                    echo "Submitting job $JOBNAME with the following parameters:"
                    echo "Dataset: $DATASET"
                    # echo "Partition Method: $PARTITION"
                    echo "Number of Nodes: $NODES"
                    # echo "Number of Sampler Processes: $SAMPLER_PROCESSES"
                    # echo "Number of Trainers: $TRAINERS"
                    # echo "Using $BACKEND backend"
                    echo "Summary file: $SUMMARYFILE"
                    # echo "Running script: $SCRIPT"
                    # echo "IP Config file: $IP_CONFIG_FILE"
                    # echo "Profile Directory: $PROFILE_DIR"
                    echo "Eviction Period: $EVICTION_PERIOD"
                    echo "Prefetch Fraction: $PREFETCH_FRACTION"
                    echo "Alpha: $ALPHA"
                    echo "Time: $TIME"
                    # if mode is gpu
                    if [ "$MODE" == "gpu" ]; then
                        CMD="sbatch -N $NODES -q $QUEUE --job-name $JOBNAME -o $OUTFILE -e $ERRFILE --time=$TIME $SCRIPT  $DATASET $PARTITION \
                        $NODES $SAMPLER_PROCESSES $SUMMARYFILE $IP_CONFIG_FILE $TRAINERS $BACKEND $PROFILE $EVICTION_PERIOD $PREFETCH_FRACTION $ALPHA $HIT_RATE $MODEL"
                    elif [ "$MODE" == "cpu" ]; then
                        CMD="sbatch -N $NODES -q $QUEUE --job-name $JOBNAME -o $OUTFILE -e $ERRFILE --time=$TIME $SCRIPT $DATASET $PARTITION \
                        $NODES $SAMPLER_PROCESSES $SUMMARYFILE $IP_CONFIG_FILE $BACKEND $PROFILE_DIR $TRAINERS $EVICTION_PERIOD $PREFETCH_FRACTION $ALPHA $HIT_RATE $MODEL"
                    fi               
                    # Submit the job
                    eval $CMD
                done
            done
        done
    done
done

#!/bin/bash
#SBATCH -A m1302
#SBATCH --constraint=cpu
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=asarkar1@iastate.edu

ulimit -c unlimited
ulimit -v unlimited
echo $LD_LIBRARY_PATH
source activate dgl-dev-gpu-117
module load cudatoolkit/11.7
echo $LD_LIBRARY_PATH
PYTHON_PATH=$(which python)
echo "Python Path: $PYTHON_PATH"
# source activate dgl-nightly

echo "Setting Project path..."

DATASET_NAME=$1
PARTITION_METHOD=$2
NUM_NODES=$3
SAMPLER_PROCESSES=$4
SUMMARYFILE=$5
IP_CONFIG_FILE=$6
BACKEND=$7
PROFILE_DIR=$8
TRAINERS=$9
EVICTION_PERIOD=${10}
PREFETCH_FRACTION=${11}
ALPHA=${12}
HIT_RATE=${13}
MODEL=${14}
JOBID=$SLURM_JOB_ID

DATA_DIR="/pscratch/sd/s/sark777/Distributed_DGL/dataset"
PROJ_PATH="/global/homes/s/sark777/Distributed_DGL/src/prefetch/prefetchv4"
PARTITION_DIR="/pscratch/sd/s/sark777/Distributed_DGL/partitions/${PARTITION_METHOD}/${DATASET_NAME}/${NUM_NODES}_parts/${DATASET_NAME}.json"
RPC_LOG_DIR="/global/cfs/cdirs/m4626/Distributed_DGL/dgl_ex/experiments/logs/blocktimes/prefetchv4/logs_${SLURM_JOB_ID}"
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST) # get list of nodes

# append job id to SUMMARYFILE and .txt
SUMMARYFILE="${SUMMARYFILE}_${SLURM_JOB_ID}.txt"
# append job id to IP_CONFIG_FILE
IP_CONFIG_FILE="${IP_CONFIG_FILE}_${SLURM_JOB_ID}.txt"

# write all parameters to summary file
echo "Assigned Nodes: $NODELIST" >> $SUMMARYFILE
echo "Dataset: $DATASET_NAME" >> $SUMMARYFILE
echo "Partition Method: $PARTITION_METHOD" >> $SUMMARYFILE
echo "Number of Nodes: $NUM_NODES" >> $SUMMARYFILE
echo "Using $BACKEND backend" >> $SUMMARYFILE
echo "Number of Sampler Processes: $SAMPLER_PROCESSES" >> $SUMMARYFILE
echo "Number of Trainers: $TRAINERS" >> $SUMMARYFILE
echo "Total GPUs: $TOTAL_GPUS" >> $SUMMARYFILE
echo "Total Nodes: $NUM_NODES" >> $SUMMARYFILE
echo "GPUs per Node: $GPUS_PER_NODE" >> $SUMMARYFILE
echo "Data Directory: $DATA_DIR" >> $SUMMARYFILE
echo "Project Path: $PROJ_PATH" >> $SUMMARYFILE
echo "Partition Directory: $PARTITION_DIR" >> $SUMMARYFILE
echo "IP Config File: $IP_CONFIG_FILE" >> $SUMMARYFILE
echo "Profile Directory: $PROFILE_DIR" >> $SUMMARYFILE
echo "Eviction Period: $EVICTION_PERIOD" >> $SUMMARYFILE
echo "Prefetch Fraction: $PREFETCH_FRACTION" >> $SUMMARYFILE
echo "Alpha: $ALPHA" >> $SUMMARYFILE
echo "Hit Rate: $HIT_RATE" >> $SUMMARYFILE
echo "Start Time: $(date +'%T.%N')" >> $SUMMARYFILE
echo "" >> $SUMMARYFILE

echo "Generating ip_config.txt..."
: > $IP_CONFIG_FILE  # Empty the file if it exists

for node in $NODELIST; do
    echo "Getting first 10.249.x.x IP address for node: $node"

    # Use srun to execute 'ip addr show' on each node and extract the first 10.249.x.x IP
    first_ip=$(srun --nodes=1 --nodelist=$node ip addr show | grep 'inet 10.249' | awk '{print $2}' | cut -d'/' -f1 | head -n 1)

    # Check if an IP address was found and append it to the ipconfig file
    if [ ! -z "$first_ip" ]; then
        echo $first_ip >> $IP_CONFIG_FILE
    else
        echo "No 10.249.x.x IP found for $node" >&2
    fi
done

# Print the contents of the ipconfig file
cat $IP_CONFIG_FILE

# assert that the number of lines in ip_config.txt is equal to NUM_NODES
NUM_IPS=$(wc -l < $IP_CONFIG_FILE)
if [ "$NUM_IPS" -ne "$NUM_NODES" ]; then
    echo "Number of IPs ($NUM_IPS) does not match number of nodes ($NUM_NODES)"
    exit 1
fi

# Delete the line with 127.0.1.1
# sed -i '/^127\.0\.1\.1/d' $IP_CONFIG_FILE
echo "JOBID: $JOBID"

# if alpha and period is 0, set eviction to False
if [ "$EVICTION_PERIOD" -eq 0 ] && [ "$ALPHA" -eq 0 ]; then
    EVICTION=False
else
    EVICTION=True
fi

# get total number of cores on the node (multiply by number of sockets)
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
SOCKETS=$(lscpu | grep "Socket(s)" | awk '{print $2}')
TOTAL_CORES=$(($CORES_PER_SOCKET * $SOCKETS))

CORES_PER_TRAINER=$(($TOTAL_CORES / $TRAINERS))
OMP_THREADS=$(($CORES_PER_TRAINER / 2))
# numba threads = OMP threads - 1
NUMBA_THREADS=$(($OMP_THREADS)) # 1 for main thread and rest for numba threads

echo "Total Cores: $TOTAL_CORES"
echo "Cores per Trainer: $CORES_PER_TRAINER"
echo "OMP Threads: $OMP_THREADS"
echo "Numba Threads: $NUMBA_THREADS"

# echo "Setting num_omp_threads to 64"
if [ "$MODEL" == "sage" ]; then
    $PYTHON_PATH $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $TRAINERS \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config  $IP_CONFIG_FILE \
    --num_omp_threads $OMP_THREADS \
    "$PYTHON_PATH node_classification_halo_nodes_degree_fetch.py --graph_name $DATASET_NAME \
    --backend $BACKEND \
    --ip_config $IP_CONFIG_FILE --num_epochs 100 --batch_size 2000 \
    --summary_filepath $SUMMARYFILE \
    --profile_dir $PROFILE_DIR \
    --prefetch_fraction $PREFETCH_FRACTION \
    --eviction_period $EVICTION_PERIOD \
    --alpha $ALPHA \
    --rpc_log_dir $RPC_LOG_DIR \
    --eviction $EVICTION \
    --num_trainer_threads $OMP_THREADS \
    --num_numba_threads $NUMBA_THREADS \
    --hit_rate_flag $HIT_RATE \
    --model $MODEL"
fi
if [ "$MODEL" == "gat" ]; then
    $PYTHON_PATH $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $TRAINERS \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config  $IP_CONFIG_FILE \
    --num_omp_threads $OMP_THREADS \
    "$PYTHON_PATH node_classification_halo_nodes_degree_fetch.py --graph_name $DATASET_NAME \
    --backend $BACKEND \
    --ip_config $IP_CONFIG_FILE --num_epochs 100 --batch_size 2000 \
    --summary_filepath $SUMMARYFILE \
    --profile_dir $PROFILE_DIR \
    --prefetch_fraction $PREFETCH_FRACTION \
    --eviction_period $EVICTION_PERIOD \
    --alpha $ALPHA \
    --rpc_log_dir $RPC_LOG_DIR \
    --eviction $EVICTION \
    --num_trainer_threads $OMP_THREADS \
    --num_numba_threads $NUMBA_THREADS \
    --hit_rate_flag $HIT_RATE \
    --model $MODEL \
    --num_heads 2"
fi
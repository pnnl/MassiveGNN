#!/bin/bash
#SBATCH -A m1302
#SBATCH -t 00:15:00
#SBATCH --constraint=gpu
#SBATCH -q debug
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=asarkar1@iastate.edu

echo $LD_LIBRARY_PATH
module load cudatoolkit/11.7
echo $LD_LIBRARY_PATH
source activate dgl-dev-gpu-117
PYTHON_PATH=$(which python)

DATASET_NAME=$1
PARTITION_METHOD=$2
NUM_NODES=$3
SAMPLER_PROCESSES=$4
SUMMARYFILE=$5
IP_CONFIG_FILE=$6
GPUS_PER_NODE=$7 # number of GPUs per node; also number of trainers
BACKEND=$8
PROFILE=$9
MODEL=${10}
TOTAL_GPUS=$(($GPUS_PER_NODE * $NUM_NODES)) # total number of GPUs
JOBID=$SLURM_JOB_ID

echo "MODEL: $MODEL"
DATA_DIR="/pscratch/sd/s/sark777/Distributed_DGL/dataset"
PROJ_PATH="/global/u1/s/sark777/MassiveGNN"
PARTITION_DIR="/pscratch/sd/s/sark777/Distributed_DGL/partitions/${PARTITION_METHOD}/${DATASET_NAME}/${NUM_NODES}_parts/${DATASET_NAME}.json"
PROFILE_DIR="/pscratch/sd/s/sark777/Distributed_DGL/rpc_cprofile/${PARTITION_METHOD}/${DATASET_NAME}/${NUM_NODES}_parts"
RPC_LOG_DIR="/global/cfs/cdirs/m4626/Distributed_DGL/dgl_ex/experiments/logs/blocktimes/baseline/logs_${SLURM_JOB_ID}"

# create PROFILE_DIR if it doesn't exist
if [ ! -d "$PROFILE_DIR" ]; then
  mkdir -p $PROFILE_DIR
fi
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
echo "Number of Sampler Processes: $SAMPLER_PROCESSES" >> $SUMMARYFILE
echo "Total GPUs: $TOTAL_GPUS" >> $SUMMARYFILE
echo "Total Nodes: $NUM_NODES" >> $SUMMARYFILE
echo "GPUs per Node: $GPUS_PER_NODE" >> $SUMMARYFILE
echo "Data Directory: $DATA_DIR" >> $SUMMARYFILE
echo "Project Path: $PROJ_PATH" >> $SUMMARYFILE
echo "Partition Directory: $PARTITION_DIR" >> $SUMMARYFILE
echo "IP Config File: $IP_CONFIG_FILE" >> $SUMMARYFILE
echo "Profile: $PROFILE" >> $SUMMARYFILE
# log current time in hh:mm:ss format
echo "Start Time: $(date +"%T")" >> $SUMMARYFILE
echo "Model: $MODEL" >> $SUMMARYFILE
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
sed -i '/^127\.0\.1\.1/d' $IP_CONFIG_FILE
echo "JOBID: $JOBID"


# if model is sage
if [ "$MODEL" == "sage" ]; then
    echo "Running sage..."
    $PYTHON_PATH $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $GPUS_PER_NODE \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config  $IP_CONFIG_FILE \
    "$PYTHON_PATH baseline/node_classification.py --graph_name $DATASET_NAME \
    --backend $BACKEND \
    --ip_config $IP_CONFIG_FILE --num_epochs 100 --batch_size 2000 \
    --num_gpus $GPUS_PER_NODE --summary_filepath $SUMMARYFILE \
    --profile_dir $PROFILE_DIR \
    --rpc_log_dir $RPC_LOG_DIR"
fi

if [ "$MODEL" == "gat" ]; then
    echo "Running gat..."
    $PYTHON_PATH $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $GPUS_PER_NODE \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config  $IP_CONFIG_FILE \
    "$PYTHON_PATH baseline/node_classification.py --graph_name $DATASET_NAME \
    --backend $BACKEND \
    --ip_config $IP_CONFIG_FILE --num_epochs 100 --batch_size 2000 \
    --num_gpus $GPUS_PER_NODE --summary_filepath $SUMMARYFILE \
    --profile_dir $PROFILE_DIR \
    --rpc_log_dir $RPC_LOG_DIR \
    --num_heads 2"
fi



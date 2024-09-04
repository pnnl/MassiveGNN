#!/bin/bash
#SBATCH --job-name=massivegnn
#SBATCH --partition=g4q  # Use a partition with GPU instances if your code requires GPU
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

# Ensure Docker is running on all nodes
srun --nodes=$SLURM_JOB_NUM_NODES sudo systemctl start docker

DATASET_NAME=$1
PARTITION_METHOD=$2
NUM_NODES=$3
SAMPLER_PROCESSES=$4
SUMMARYFILE=$5
IP_CONFIG_FILE=$6
GPUS_PER_NODE=$7 # number of GPUs per node; also number of trainers
BACKEND=$8
EVICTION_PERIOD=$9
PREFETCH_FRACTION=${10}
ALPHA=${11}
HIT_RATE=${12}
MODEL=${13}
DATA_DIR=${14}
PROJ_PATH=${15}
PARTITION_DIR=${16}
AWS_ACCESS_KEY_ID=${17}
AWS_SECRET_ACCESS_KEY=${18}
AWS_SESSION_TOKEN=${19}


TOTAL_GPUS=$(($GPUS_PER_NODE * $NUM_NODES)) # total number of GPUs
JOBID=$SLURM_JOB_ID

NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST) # get list of nodes
SUMMARYFILE="${SUMMARYFILE}_${SLURM_JOB_ID}.txt" # append job id to SUMMARYFILE and .txt
IP_CONFIG_FILE="${IP_CONFIG_FILE}_${SLURM_JOB_ID}.txt" # append job id to IP_CONFIG_FILE

S3_BUCKET="s3://massivegnn"
S3_PARTITION_PATH="$S3_BUCKET/dgl-partitions"

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
echo "Eviction Period: $EVICTION_PERIOD" >> $SUMMARYFILE
echo "Prefetch Fraction: $PREFETCH_FRACTION" >> $SUMMARYFILE
echo "Alpha: $ALPHA" >> $SUMMARYFILE
echo "Hit Rate: $HIT_RATE" >> $SUMMARYFILE
echo "Start Time: $(date +'%T.%N')" >> $SUMMARYFILE
echo "" >> $SUMMARYFILE

echo "Generating ip_config.txt..."
: > $IP_CONFIG_FILE  # Empty the file if it exists
for node in $NODELIST; do
    first_ip=$(srun --nodes=1 --nodelist=$node hostname -I | awk '{print $1}')
    if [ ! -z "$first_ip" ]; then
        echo $first_ip >> $IP_CONFIG_FILE
    else
        echo "No IP found for $node" >&2
    fi
done

NUM_IPS=$(wc -l < $IP_CONFIG_FILE)
if [ "$NUM_IPS" -ne "$NUM_NODES" ]; then
    echo "Number of IPs ($NUM_IPS) does not match number of nodes ($NUM_NODES)"
    exit 1
fi

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
TRAINERS=$GPUS_PER_NODE

CORES_PER_TRAINER=$(($TOTAL_CORES / $TRAINERS))
OMP_THREADS=$(($CORES_PER_TRAINER)) # OMP threads = number of cores per trainer
# numba threads = OMP threads - 1
NUMBA_THREADS=$(($OMP_THREADS)) # all cores to numba threads as gpu is used for training

echo "Total Cores: $TOTAL_CORES"
echo "Cores per Trainer: $CORES_PER_TRAINER"
echo "OMP Threads: $OMP_THREADS"
echo "Numba Threads: $NUMBA_THREADS"

echo "JOBID: $JOBID"

export $AWS_ACCESS_KEY_ID
export $AWS_SECRET_ACCESS_KEY
export $AWS_SESSION_TOKEN

# Pull data from S3 outside Docker
aws s3 cp --recursive $S3_PARTITION_PATH/${DATASET_NAME}/${NUM_NODES}_parts $PROJ_PATH/partitions/${DATASET_NAME}/${NUM_NODES}_parts/

echo "Stopping and removing existing containers..."
srun --nodes=$SLURM_JOB_NUM_NODES bash -c "
    existing_container=\$(docker ps -aq -f name=dgl_container)
    if [ ! -z \"\$existing_container\" ]; then
        docker stop \$existing_container
        docker rm \$existing_container
    fi
"

# Start the Docker container as a daemon on all nodes
echo "Starting Docker container as a daemon on all nodes..."
srun --nodes=$SLURM_JOB_NUM_NODES bash -c "
    docker run --gpus all \
   --network=host \
   --ipc=host \
   --privileged \
   -v /home/ec2-user:/home/ec2-user \
   -v /home/ec2-user/.ssh:/root/.ssh \
   -w /home/ec2-user/MassiveGNN \
   --name dgl_container \
   -d \
   nvcr.io/nvidia/dgl:24.07-py3 \
   tail -f /dev/null
"

# Ensure the container is running on all nodes
srun --nodes=$SLURM_JOB_NUM_NODES bash -c "
    if ! docker ps | grep -q dgl_container; then
        echo 'Container dgl_container is not running on node \$(hostname)' >&2
        exit 1
    fi
"

if [ "$MODEL" == "sage" ]; then
    echo "Running SAGE model..."
    python3 $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $GPUS_PER_NODE \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config  $IP_CONFIG_FILE \
    --num_omp_threads $OMP_THREADS \
    --docker_container dgl_container \
    "python3 massivegnn/main.py --graph_name $DATASET_NAME \
    --backend $BACKEND \
    --ip_config $IP_CONFIG_FILE --num_epochs 100 --batch_size 2000 \
    --num_gpus $GPUS_PER_NODE --summary_filepath $SUMMARYFILE \
    --prefetch_fraction $PREFETCH_FRACTION --eviction_period $EVICTION_PERIOD --alpha $ALPHA \
    --eviction $EVICTION \
    --num_numba_threads $NUMBA_THREADS \
    --hit_rate_flag $HIT_RATE \
    --model $MODEL"
fi

if [ "$MODEL" == "gat" ]; then
    echo "Running GAT model..."
    $PYTHON_PATH $PROJ_PATH/launch.py \
    --workspace $PROJ_PATH \
    --num_trainers $GPUS_PER_NODE \
    --num_samplers $SAMPLER_PROCESSES \
    --num_servers 1 \
    --part_config $PARTITION_DIR \
    --ip_config  $IP_CONFIG_FILE \
    --num_omp_threads $OMP_THREADS \
    --docker_container dgl_container \
    "$PYTHON_PATH massivegnn/main.py --graph_name $DATASET_NAME \
    --backend $BACKEND \
    --ip_config $IP_CONFIG_FILE --num_epochs 100 --batch_size 2000 \
    --num_gpus $GPUS_PER_NODE --summary_filepath $SUMMARYFILE \
    --prefetch_fraction $PREFETCH_FRACTION --eviction_period $EVICTION_PERIOD --alpha $ALPHA \
    --eviction $EVICTION \
    --num_numba_threads $NUMBA_THREADS \
    --hit_rate_flag $HIT_RATE \
    --model $MODEL \
    --num_heads 2"
fi

# Stop the Docker container after the job is done
echo "Stopping Docker container..."
srun --nodes=$SLURM_JOB_NUM_NODES bash -c "docker stop dgl_container"

echo "Job completed successfully."
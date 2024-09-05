import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from collections import defaultdict
import socket
import dgl
import numpy as np
import torch as th
import tqdm
import cProfile
import utils
from trainer import Trainer
import datetime

def main(args):
    """
    Main function.
    """
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(args.ip_config)
    print(f"{host_name}: Initializing PyTorch process group.")
    th.distributed.init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=5400))
    local_rank = args.local_rank
    # get pytorch's local rank
    print(f"Local rank: {args.local_rank}")
    utils.set_numa_affinity(local_rank)
    print(f"CPU affinity of process {os.getpid()} rank {local_rank}: {os.sched_getaffinity(0)}")
    print(f"{host_name}: Initializing DistGraph.")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print("Graph Obj g.ndata:", g.ndata)

    # Split train/val/test IDs for each trainer.
    pb = g.get_partition_book()
    print(f"Partition book metadata of {host_name}", pb.metadata())
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy() # get local node ids
    num_train_local = len(np.intersect1d(train_nid.numpy(), local_nid)) 
    num_val_local = len(np.intersect1d(val_nid.numpy(), local_nid))
    num_test_local = len(np.intersect1d(test_nid.numpy(), local_nid))
    print(
        f"part {g.rank()}, train: {len(train_nid)} (local: {num_train_local}), "
        f"val: {len(val_nid)} (local: {num_val_local}), "
        f"test: {len(test_nid)} (local: {num_test_local})"
    )

    del local_nid
    if args.num_gpus == 0:
        device = th.device("cpu")
        print("Using CPU.")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        print(f"Using GPU {dev_id}.")
    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.ndata["labels"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Pack data.
    in_feats = g.ndata["features"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    trainer = Trainer(args, device, data, utils.get_halos(g, pb, train_nid, args.num_layers))
    print(f"Rank {g.rank()} Trainer and Prefetcher Initialized.")

    # Train and evaluate.
    (epoch_time, test_acc, forward_time, backward_time, update_time, sample_time, eval_time, 
     hit_rate, miss_rate, alpha, period, threshold, absolute_total_time,
     prefetch_time) = trainer.run()

    print(
        f"Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f} "
        f"| TestAccuracy {test_acc:.4f} | ForwardTime {forward_time:.4f}"
        f"| BackwardTime {backward_time:.4f} | UpdateTime {update_time:.4f}"
        f" | SampleTime {sample_time:.4f} | EvalTime {eval_time:.4f}"
    )

    # calculate the mean epoch time accross all processes
    epoch_time_tensor = utils.calculate_mean(epoch_time, device)
    forward_time_tensor = utils.calculate_mean(forward_time, device)
    backward_time_tensor = utils.calculate_mean(backward_time, device)
    update_time_tensor = utils.calculate_mean(update_time, device)
    sample_time_tensor = utils.calculate_mean(sample_time, device)
    eval_time_tensor = utils.calculate_mean(eval_time, device)
    test_acc_tensor = utils.calculate_mean(test_acc, device) 
    total_epoch_time_tensor = utils.sum(absolute_total_time['epoch_time'], device)
    # Generate the summary string for the current rank
    summary_str = (
        "\n"
            f"Rank {g.rank()} | TotalEpochTime {absolute_total_time['epoch_time']:.4f}s"
            f"| HitRate {hit_rate:.4f} | MissRate {miss_rate:.4f}"
            f"| ForwardTime {absolute_total_time['forward_time']:.4f}s"
            f"| BackwardTime {absolute_total_time['backward_time']:.4f}s"
            f"| UpdateTime {absolute_total_time['update_time']:.4f}s"
            f"| FirstMinibatchSampleTime {absolute_total_time['first_minibatch_sample_time']:.4f}s"
            f"| SampleTime {absolute_total_time['sample_time']:.4f}s"
            f"| WaitForThreadTime {absolute_total_time['wait_for_thread_time']:.4f}s"
            f"| EvalTime {absolute_total_time['eval_time']:.4f}s"
            f"| EpochTime80Percent {absolute_total_time['epoch_time_80_percent']:.4f}s"
            f"| PrefetchComputeTime {prefetch_time['prefetch_compute_time']:.4f}s"
            f"| EvictionTime {prefetch_time['eviction_time']:.4f}s"
            f"| RPCTime {prefetch_time['rpc_time']:.4f}s"
            "\n"
    )

    # Prepare to gather the summary strings to rank 0
    gathered_summaries = None

    if g.rank() == 0:
        # Rank 0 will gather summaries from all ranks, including its own
        gathered_summaries = [None] * th.distributed.get_world_size()

    # Gather the summary strings from all ranks to rank 0
    th.distributed.gather_object(
        summary_str,  # The object to send (for all ranks)
        gathered_summaries,  # Only for rank 0, where gathered objects will be stored
        dst=0  # Gather to rank 0
    )

    # Rank 0 writes all summaries to the file
    if g.rank() == 0:
        with open(args.summary_filepath, "a") as f:
            for summary in gathered_summaries:
                if summary is not None:
                    print(f"Rank 0 | Writing gathered summaries to {args.summary_filepath}")
                    f.write(summary)


    # print the final summary
    if th.distributed.get_rank() == 0:
        print("Average training time across processes: {:.4f} seconds".format(epoch_time_tensor))
        print("Average forward time across processes: {:.4f} seconds".format(forward_time_tensor))
        print("Average backward time across processes: {:.4f} seconds".format(backward_time_tensor))
        print("Average update time across processes: {:.4f} seconds".format(update_time_tensor))
        print("Average sample time across processes: {:.4f} seconds".format(sample_time_tensor))
        print("Average eval time across processes: {:.4f} seconds".format(eval_time_tensor))
        print("Average test accuracy across processes: {:.4f}".format(test_acc_tensor))
        
        # write the summary to a args.summary_filepath
        with open(args.summary_filepath, "a") as f:
            f.write(f"alpha: {alpha}, period: {period}, threshold: {threshold}")
            f.write(
                "\n"
                "\n"
                f"Summary of node classification({args.model}): GraphName, prefetch_fraction: {args.prefetch_fraction}, "
                f"{args.graph_name} | TrainEpochTime(mean) {epoch_time_tensor:.4f} | TotalEpochTime {total_epoch_time_tensor:.4f}"
                f"| TestAccuracy {test_acc_tensor:.4f} | ForwardTime {forward_time_tensor:.4f}"
                f"| BackwardTime {backward_time_tensor:.4f} | UpdateTime {update_time_tensor:.4f}"
                f"| SampleTime+Data_Copy {sample_time_tensor:.4f} | EvalTime {eval_time_tensor:.4f}"
                "\n"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE.")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="the number of GPU device. Use 0 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local-rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    parser.add_argument(
        "--summary_filepath", type=str, help="path to save summary file"
    )
    parser.add_argument(
        "--prefetch_fraction", type=float, default=0.5, help="prefetch fraction"
    )
    parser.add_argument(
        "--eviction_period", type=int, default=25, help="eviction period"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="alpha"
    )
    parser.add_argument(
        "--num_numba_threads", type=int, default=1, help="number of numba threads"
    )
    parser.add_argument(
        "--num_trainer_threads", type=int, default=1, help="number of trainer threads"
    )
    parser.add_argument(
        "--hit_rate_flag", type=utils.str2bool, default=False, help="Enable or disable hit rate flag. Accepts: True or False"
    )
    parser.add_argument(
        "--model", type=str, default="sage", help="Model to use for training. Accepts: graphsage or gat"
    )
    parser.add_argument("--eviction", type=utils.str2bool, default=True, help="Enable or disable eviction. Accepts: True or False")
    parser.add_argument("--num_heads", type=int, default=0, help="Number of attention heads")
    args = parser.parse_args()
    if args.model == "gat":
        assert args.num_heads > 0, "Number of attention heads must be greater than 0"
    
    if args.model not in ["sage", "gat"]:
        raise ValueError("Model not supported. Please choose either graphsage or gat.")
    
    main(args)
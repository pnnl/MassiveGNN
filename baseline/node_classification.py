import os, sys
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import socket
import time
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import cProfile
import math
from models.graphsage import DistSAGE
from models.gat import GAT


# from scalene import scalene_profiler
def set_numa_affinity(rank):
    numa_bindings = {
        0: set(range(0, 32)).union(range(128, 160)),  # Ranks 0 covers NUMA nodes 0 and 1
        1: set(range(32, 64)).union(range(160, 192)), # Ranks 1 covers NUMA nodes 2 and 3
        2: set(range(64, 96)).union(range(192, 224)), # Ranks 2 covers NUMA nodes 4 and 5
        3: set(range(96, 128)).union(range(224, 256)),# Ranks 3 covers NUMA nodes 6 and 7
    }
    
    if rank in numa_bindings:
        cpu_ids = numa_bindings[rank]
        os.sched_setaffinity(0, cpu_ids)

def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, num_heads, device):
    """
    Evaluate the model on the validation and test set.

    Parameters
    ----------
    model : DistSAGE
        The model to be evaluated.
    g : DistGraph
        The entire graph.
    inputs : DistTensor
        The feature data of all the nodes.
    labels : DistTensor
        The labels of all the nodes.
    val_nid : torch.Tensor
        The node IDs for validation.
    test_nid : torch.Tensor
        The node IDs for test.
    batch_size : int
        Batch size for evaluation.
    device : torch.Device
        The target device to evaluate on.

    Returns
    -------
    float
        Validation accuracy.
    float
        Test accuracy.
    """
    model.eval()
    with th.no_grad():
        if args.model == "sage":
            pred = model.inference(g, inputs, batch_size, device)
        elif args.model == "gat":
            pred = model.inference(g, inputs, num_heads, device, batch_size)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def compute_acc(pred, labels):
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def run(args, device, data):
    """
    Train and evaluate DistSAGE.

    Parameters
    ----------
    args : argparse.Args
        Arguments for train and evaluate.
    device : torch.Device
        Target device for train and evaluate.
    data : Packed Data
        Packed data includes train/val/test IDs, feature dimension,
        number of classes, graph.
    """
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    # use full neighbor sampler
    sampler = dgl.dataloading.NeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(",")]
            )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    if args.model == "gat":
        model = DistGAT(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
            args.num_heads,
            F.relu,
            allow_zero_in_degree=True
        )
    elif args.model == "sage":
        model = DistSAGE(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
            F.relu,
            args.dropout,
        )
    model = model.to(device)
    if args.num_gpus == 0:
        model = th.nn.parallel.DistributedDataParallel(model)
    else:
        model = th.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = nn.CrossEntropyLoss()
    # Training loop.
    iter_tput = []
    epoch = 0
    epoch_time = []
    forward_time_list = []
    backward_time_list = []
    update_time_list = []
    sample_time_list = []
    eval_time = []
    rpc_time_list = []
    test_acc = 0.0
    num_mini_batches = math.ceil(len(train_nid) / args.batch_size)
    print("Total number of minibatches: ", num_mini_batches * args.num_epochs)
    dataloader_iter = dataloader.__iter__()
    for _ in range(args.num_epochs):
        # dgl.distributed.rpc.set_training_phase(True)
        epoch += 1
        tic = time.time()
        # Various time statistics.
        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        rpc_time = 0
        start = time.time()
        step_time = []
        with model.join():
            # if cpu
            if device == th.device("cpu"):
                dgl.utils.set_num_threads(16)
            step = 0
            while step < num_mini_batches:
                if step == num_mini_batches - 1:
                    dataloader_iter = dataloader.__iter__() # if last step, reset the dataloader for the next epoch
                tic_step = time.time()
                # dgl.distributed.rpc.set_training_phase(True)
                start = time.time() 
                input_nodes, seeds, blocks = next(dataloader_iter)         
                sample_time += time.time() - start
                start_rpc = time.time()
                batch_inputs = g.ndata["features"][input_nodes]
                rpc_time += time.time() - start_rpc
                batch_labels = g.ndata["labels"][seeds].long()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                # Move to target device.
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                # Compute loss and prediction.
                start = time.time()
                batch_pred = model(blocks, batch_inputs)
                if args.model == "gat":
                    loss = F.nll_loss(batch_pred, batch_labels)
                elif args.model == "sage":
                    loss = loss_fcn(batch_pred, batch_labels)
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end

                optimizer.step()
                update_time += time.time() - compute_end

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if (step + 1) % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    sample_speed = np.mean(iter_tput[-args.log_every :])
                    mean_step_time = np.mean(step_time[-args.log_every :])
                    print(
                        f"Part {g.rank()} | Epoch {epoch:05d} | Step {step:05d}"
                        f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}"
                        f" | Speed (samples/sec) {sample_speed:.4f}"
                        f" | GPU {gpu_mem_alloc:.1f} MB | "
                        f"Mean step time {mean_step_time:.3f} s"
                    )
                step += 1
                # start = time.time()

        toc = time.time()
        print(
            f"Part {g.rank()}, epoch: {epoch}, Epoch Time(s): {toc - tic:.4f}, "
            f"sample+data_copy: {sample_time:.4f}, rpc: {rpc_time:.4f},"
            f" forward: {forward_time:.4f},"
            f" backward: {backward_time:.4f}, update: {update_time:.4f}, "
            f"#seeds: {num_seeds}, #inputs: {num_inputs}"
        )
        epoch_time.append(toc - tic)
        forward_time_list.append(forward_time)
        backward_time_list.append(backward_time)
        update_time_list.append(update_time)
        sample_time_list.append(sample_time)
        rpc_time_list.append(rpc_time)

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            # dgl.distributed.rpc.set_training_phase(False)
            start = time.time()
            val_acc, test_acc = evaluate(
                model.module,
                g,
                g.ndata["features"],
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                args.num_heads,
                device
            )
            print(
                f"Part {g.rank()}, Val Acc {val_acc:.4f}, "
                f"Test Acc {test_acc:.4f}, time: {time.time() - start:.4f}"
            )
            eval_time.append(time.time() - start)

    # sum last 80% of epoch time
    epoch_time_80_percent = epoch_time[int(len(epoch_time)*0.2):]
    # store time in a dict
    absolute_total_time = {
        'epoch_time': np.sum(epoch_time), 'forward_time': np.sum(forward_time_list), 'rpc_time': np.sum(rpc_time_list),
        'backward_time': np.sum(backward_time_list), 'update_time': np.sum(update_time_list),
        'sample_time': np.sum(sample_time_list), 'eval_time': np.sum(eval_time), 'epoch_time_80_percent': np.sum(epoch_time_80_percent)}
    return (np.mean(epoch_time), test_acc, np.mean(forward_time_list), np.mean(rpc_time_list), 
            np.mean(backward_time_list), np.mean(update_time_list), np.mean(sample_time_list), 
            np.mean(eval_time), absolute_total_time)


def main(args):
    """
    Main function.
    """
    # pr = cProfile.Profile()
    # pr.enable()
    # get pid
    # pid = os.getpid()
    # current_process = psutil.Process(pid)
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(args.ip_config)
    print(f"{host_name}: Initializing PyTorch process group.")
    th.distributed.init_process_group(backend=args.backend)
    local_rank = args.local_rank
    # get pytorch's local rank
    print(f"Local rank: {args.local_rank}")
    set_numa_affinity(local_rank)
    print(f"CPU affinity of process {os.getpid()} rank {local_rank}: {os.sched_getaffinity(0)}")
    print(f"{host_name}: Initializing DistGraph.")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(f"Rank of {host_name}: {g.rank()}")
    # dgl.distributed.rpc.set_log_dir(args.rpc_log_dir)
    # Split train/val/test IDs for each trainer.
    pb = g.get_partition_book()
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
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
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
        del labels
    print(f"Number of classes: {n_classes}")

    # Pack data.
    in_feats = g.ndata["features"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g

    # Train and evaluate.
    # epoch_time, test_acc = run(args, device, data)
    (epoch_time, test_acc, forward_time, rpc_time, backward_time, update_time, 
    sample_time, eval_time, absolute_total_time) = run(args, device, data)
    print(
        f"Summary of node classification({args.model}): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f}",
        f"| TestAccuracy {test_acc:.4f}",
        f"| ForwardTime {forward_time:.4f}" 
        f"| BackwardTime {backward_time:.4f} | UpdateTime {update_time:.4f}"
        f"| SampleTime {sample_time:.4f} | EvalTime {eval_time:.4f}"
    )

    # calculate the mean epoch time accross all processes
    epoch_time_tensor = calculate_mean(epoch_time, device)
    forward_time_tensor = calculate_mean(forward_time, device)
    rpc_time_tensor = calculate_mean(rpc_time, device)
    backward_time_tensor = calculate_mean(backward_time, device)
    update_time_tensor = calculate_mean(update_time, device)
    sample_time_tensor = calculate_mean(sample_time, device)
    eval_time_tensor = calculate_mean(eval_time, device)
    test_acc_tensor = calculate_mean(test_acc, device)
    total_epoch_time_tensor = sum(absolute_total_time['epoch_time'], device)

    # Write individual rank's total epoch time to args.summary_filepath
    with open(args.summary_filepath, "a") as f:
        f.write(
            "\n"
            f"Rank {g.rank()} | TotalEpochTime {absolute_total_time['epoch_time']:.4f}s"
            f"| ForwardTime {absolute_total_time['forward_time']:.4f}s"
            f"| RpcTime {absolute_total_time['rpc_time']:.4f}s"
            f"| BackwardTime {absolute_total_time['backward_time']:.4f}s"
            f"| UpdateTime {absolute_total_time['update_time']:.4f}s"
            f"| SampleTime {absolute_total_time['sample_time']:.4f}s"
            f"| EvalTime {absolute_total_time['eval_time']:.4f}s"
            f"| EpochTime80Percent {absolute_total_time['epoch_time_80_percent']:.4f}s"
            "\n"
        )
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
            f.write(
                "\n"
                f"Summary of node classification(GraphSAGE): GraphName "
                f"{args.graph_name} | TrainEpochTime(mean) {epoch_time_tensor:.4f} | TotalEpochTime {total_epoch_time_tensor:.4f}"
                f"| TestAccuracy {test_acc_tensor:.4f} | ForwardTime {forward_time_tensor:.4f}"
                f"| BackwardTime {backward_time_tensor:.4f} | UpdateTime {update_time_tensor:.4f}"
                f" | SampleTime+Data_Copy {sample_time_tensor:.4f} | EvalTime {eval_time_tensor:.4f}"
                "\n"
            )
    
    # print(f"Memory usage of {host_name}: {current_process.memory_info().rss / 1024 ** 2} MB")
    # rank = g.rank()
    # pr.disable()
    # pr.dump_stats(os.path.join(args.profile_dir, f'cpu_{rank}.prof'))
    # # - for text dump
    # print(f"Rank {rank} dumping stats to cpu_{rank}.txt")
    # with open(os.path.join(args.profile_dir, f'cpu_{rank}.txt'), 'w') as output_file:
    #     sys.stdout = output_file
    #     pr.print_stats(sort='time')
    #     sys.stdout = sys.__stdout__

def calculate_mean(param, device):
    param_tensor = th.tensor(param).to(device)
    th.distributed.all_reduce(param_tensor, op=th.distributed.ReduceOp.SUM)
    param_tensor /= th.distributed.get_world_size()
    return param_tensor

def sum(param, device):
    param_tensor = th.tensor(param).to(device)
    th.distributed.all_reduce(param_tensor, op=th.distributed.ReduceOp.SUM)
    return param_tensor

def percentage(part, whole):
    return round(100 * float(part)/float(whole), 2)


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
        "--profile_dir", type=str, help="path to save profile file"
    )
    parser.add_argument(
        "--rpc_log_dir", type=str, help="path to save rpc log file"
    )
    parser.add_argument(
        "--model", type=str, default="sage", help="Model to use for training. Accepts: graphsage or gat"
    )
    parser.add_argument(
        "--num_heads" , type=int, default=1, help="Number of attention heads."
    )
    args = parser.parse_args()
    if args.model == "gat":
        assert args.num_heads > 0, "Number of attention heads must be greater than 0"
    
    if args.model not in ["sage", "gat"]:
        raise ValueError("Model not supported. Please choose either graphsage or gat.")
    
    print(f"Arguments: {args}")
    main(args)
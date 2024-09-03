
import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prefetch.prefetch import Prefetch
from prefetch.prefetch_efficient import PrefetchEfficient
from models.graphsage import DistSAGE
from models.gat import GAT
import utils
from concurrent.futures import ThreadPoolExecutor
import queue as q
import math
import time
import numpy as np
import torch as th
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pympler import muppy, summary
import yappi

class Trainer:
    def __init__(self, args, device, data, halo_nodes):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.args = args
        self.device = device
        self.data = data
        self.halo_nodes = halo_nodes
        self.train_nid, self.val_nid, self.test_nid, self.in_feats, self.n_classes, self.g = self.data
        if args.graph_name == "ogbn-papers100M":
            print("Using efficient prefetcher")
            self.prefetcher = PrefetchEfficient(self.g, self.args.prefetch_fraction, self.args.eviction_period, self.args.alpha, self.halo_nodes, self.args.num_layers, self.train_nid, self.args.num_numba_threads, self.args.eviction, self.device, self.args.hit_rate_flag)
        else:
            self.prefetcher = Prefetch(self.g, self.args.prefetch_fraction, self.args.eviction_period, self.args.alpha, self.halo_nodes, self.args.num_layers, self.train_nid, self.args.num_numba_threads, self.args.eviction, self.device, self.args.hit_rate_flag)
        self.prefetcher.degree_based_prefetch()
        self.sampler = dgl.dataloading.NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(",")]
        )
        self.dataloader = dgl.dataloading.DistNodeDataLoader(
            self.g,
            self.train_nid,
            self.sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        if args.model == "sage":
            self.model = DistSAGE(
                self.in_feats,
                self.args.num_hidden,
                self.n_classes,
                self.args.num_layers,
                F.relu,
                self.args.dropout,
            )
        elif args.model == "gat":
            self.model = GAT(
                self.in_feats,
                self.args.num_hidden,
                self.n_classes,
                self.args.num_layers,
                self.args.num_heads,
                F.relu
            )
        self.loss_fcn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.num_mini_batches = math.ceil(len(self.train_nid) / self.args.batch_size)
        self.next_batch_inputs = q.Queue()
        self.next_batch_labels = q.Queue()
        self.next_batch_blocks = q.Queue()  
        
        print("Eviction: ", self.args.eviction)
        if not self.args.eviction:
            print("No eviction")

        print(f"Total mini batches: {self.num_mini_batches * self.args.num_epochs}")
            

    def evaluate(self):
        self.model.module.eval()
        with th.no_grad():
            if self.args.model == "sage":
                pred = self.model.module.inference(self.g, self.g.ndata["features"], self.args.batch_size_eval, self.device)
            elif self.args.model == "gat":
                pred = self.model.module.inference(self.g, self.g.ndata["features"], self.args.num_heads, self.device, self.args.batch_size_eval)
        self.model.module.train()
        return utils.compute_acc(pred[self.val_nid], self.g.ndata["labels"][self.val_nid]), utils.compute_acc(
            pred[self.test_nid], self.g.ndata["labels"][self.test_nid]
        ) 

    def _get_first_minibatch(self, dataloader_iter):
        start_first_minibatch = time.time()
        input_nodes, seeds, blocks = next(dataloader_iter)
        end_first_minibatch = time.time()
        input_nodes_array = input_nodes.numpy().astype(np.int32)
        batch_inputs = th.full((len(input_nodes), self.in_feats), float('nan'))
        if not self.args.eviction:
            batch_inputs = self.prefetcher.prefetch(input_nodes_array, batch_inputs)
        else:
            batch_inputs = self.prefetcher.prefetch_with_eviction(input_nodes_array, batch_inputs)
        batch_labels = self.g.ndata["labels"][seeds].long() 
        return batch_inputs, batch_labels, blocks, end_first_minibatch - start_first_minibatch
    
    def _next_minibatch(self, dataloader_iter, g):
        try:
            start_total = time.time()
            start_fetch = time.time()
            input_nodes, seeds, blocks = next(dataloader_iter)
            end_fetch = time.time()
            fetch_time = end_fetch - start_fetch
            self.next_batch_blocks.put(blocks)
            self.next_batch_labels.put(g.ndata["labels"][seeds].long())
            start_process = time.time()
            self._fetch_and_process(input_nodes)
            end_process = time.time()
            process_time = end_process - start_process
            end_total = time.time()
            total_time = end_total - start_total
            return fetch_time, process_time, total_time, True
        except StopIteration:
            return 0, 0, 0, False
        
    def _fetch_and_process(self, input_nodes):
        start_input_nodes = time.time()
        input_nodes_array = input_nodes.numpy().astype(np.int32)
        time_input_nodes = time.time() - start_input_nodes

        start = time.time()
        batch_inputs = th.full((len(input_nodes_array), self.in_feats), float('nan'))
        batch_array_time = time.time() - start
        
        if not self.args.eviction:
            start = time.time()
            batch_inputs = self.prefetcher.prefetch(input_nodes_array, batch_inputs)
            time_prefetch = time.time() - start

            start = time.time()
            self.next_batch_inputs.put(batch_inputs)
            time_put = time.time() - start
        else:
            self.next_batch_inputs.put(self.prefetcher.prefetch_with_eviction(input_nodes_array, batch_inputs))
              
    def run(self):
        self.model = self.model.to(self.device)
        if self.args.num_gpus == 0:
            self.model = th.nn.parallel.DistributedDataParallel(self.model)
        else:
            self.model = th.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device], output_device=self.device
            )
        if self.args.model == "sage":
            self.loss_fcn = self.loss_fcn.to(self.device)
        # Training loop.
        iter_tput = []
        epoch = 0
        epoch_time = []
        forward_time_list = []
        backward_time_list = []
        update_time_list = []
        sample_time_list = []
        outer_sample_list = []
        wait_for_thread = []
        eval_time = []
        test_acc = 0.0
        dataloader_iter = self.dataloader.__iter__()
        # set the number of threads for pytorch
        for _ in range(self.args.num_epochs):
            epoch += 1
            tic = time.time()
            # Various time statistics.
            sample_time = 0
            forward_time = 0
            backward_time = 0
            update_time = 0
            num_seeds = 0
            num_inputs = 0
            thread_fetch_time = 0
            thread_process_time = 0
            wait_for_thread_time = 0
            thread_total_time = 0
            step_time = []
            start = time.time()
            with self.model.join():
                # if device is cpu, set the number of threads to 16
                if self.device == th.device("cpu"):
                    dgl.utils.set_num_threads(self.args.num_trainer_threads)
                    # print("Number of threads used by dgl: ", dgl.utils.get_num_threads(), "by torch: ", th.get_num_threads())
                step = 0
                while step < self.num_mini_batches:
                    if step == self.num_mini_batches - 1:
                        # if last step, reset the dataloader for the next epoch
                        dataloader_iter = self.dataloader.__iter__()
                    tic_step = time.time()
                    future = None
                    if step == 0 and epoch == 1:
                        # First minibatch of the first epoch.
                        batch_inputs, batch_labels, blocks, first_minibatch_sample_time = self._get_first_minibatch(dataloader_iter)
                    else:
                        start_queue = time.time()
                        batch_inputs = self.next_batch_inputs.get()
                        batch_labels = self.next_batch_labels.get()
                        blocks = self.next_batch_blocks.get()
                        take_from_queue = time.time() - start_queue
                    submit_task_start = time.time()
                    future = self.executor.submit(self._next_minibatch, dataloader_iter, self.g)
                    submit_task_time = time.time() - submit_task_start
                    num_seeds += len(blocks[-1].dstdata[dgl.NID])
                    num_inputs += len(blocks[0].srcdata[dgl.NID])
                    # Move to target device.
                    blocks = [block.to(self.device) for block in blocks]
                    batch_inputs = batch_inputs.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    # Compute loss and prediction.
                    start = time.time()
                    batch_pred = self.model(blocks, batch_inputs)
                    if self.args.model == "gat":
                        loss = F.nll_loss(batch_pred, batch_labels)
                    elif self.args.model == "sage":
                        loss = self.loss_fcn(batch_pred, batch_labels)
                    forward_end = time.time()
                    self.optimizer.zero_grad()
                    loss.backward()
                    compute_end = time.time()
                    forward_time += forward_end - start
                    backward_time += compute_end - forward_end
                    self.optimizer.step()
                    update_time += time.time() - compute_end
                    step_t = time.time() - tic_step
                    step_time.append(step_t)
                    iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                    if (step + 1) % self.args.log_every == 0:
                        acc = utils.compute_acc(batch_pred, batch_labels)
                        gpu_mem_alloc = (
                            th.cuda.max_memory_allocated() / 1000000
                            if th.cuda.is_available()
                            else 0
                        )
                        sample_speed = np.mean(iter_tput[-self.args.log_every :])
                        mean_step_time = np.mean(step_time[-self.args.log_every :])
                        print(
                            f"Part {self.g.rank()} | Epoch {epoch:05d} | Step {step:05d}"
                            f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}"
                            f" | Speed (samples/sec) {sample_speed:.4f}"
                            f" | GPU {gpu_mem_alloc:.1f} MB | "
                            f"Mean step time {mean_step_time:.3f}s"
                        )
                    # check time spent on waiting for data
                    start_thread_wait = time.time()
                    if future is not None:
                        fetch_time, process_time, total_time, has_next = future.result()
                        if not has_next:
                            # print("Breaking out of loop")
                            break
                        thread_fetch_time += fetch_time
                        thread_process_time += process_time
                        thread_total_time += total_time
                        sample_time += fetch_time
                    thread_time = time.time() - start_thread_wait
                    wait_for_thread_time += thread_time
                    step += 1
                    # start = time.time()
            toc = time.time()
            print(
                f"Part {self.g.rank()}, epoch: {epoch}, Epoch Time(s): {toc - tic:.4f}, "
                f" next_minibatch_process_time: {thread_process_time:.4f}, next_minibatch_fetch_time: {thread_fetch_time:.4f},"
                f" submit_task_time: {submit_task_time:.4f}, take_from_queue: {take_from_queue:.4f},"
                f" next_minibatch_total_time: {thread_total_time:.4f}, next_minibatch_wait_time: {wait_for_thread_time:.4f},"
                f" sample+data_copy: {sample_time:.4f}, forward: {forward_time:.4f},"
                f" backward: {backward_time:.4f}, update: {update_time:.4f}, "
                f" #seeds: {num_seeds}, #inputs: {num_inputs}, "
            )
            epoch_time.append(toc - tic)
            forward_time_list.append(forward_time)
            backward_time_list.append(backward_time)
            update_time_list.append(update_time)
            sample_time_list.append(sample_time)
            wait_for_thread.append(wait_for_thread_time)

            if epoch % self.args.eval_every == 0 or epoch == self.args.num_epochs:
                start = time.time()
                val_acc, test_acc = self.evaluate()
                print(
                    f"Part {self.g.rank()}, Val Acc {val_acc:.4f}, "
                    f"Test Acc {test_acc:.4f}, time: {time.time() - start:.4f}"
                )
                eval_time.append(time.time() - start)
        print("Total time prefetch was called: ", self.prefetcher.counter)
        self.prefetcher.close() 
        # sum last s80% of epoch time
        epoch_time_80_percent = epoch_time[int(len(epoch_time)*0.2):]
        # store time in a dict
        absolute_total_time = {
            'epoch_time': np.sum(epoch_time), 
            'forward_time': np.sum(forward_time_list),
            'backward_time': np.sum(backward_time_list), 
            'update_time': np.sum(update_time_list),
            'first_minibatch_sample_time': first_minibatch_sample_time, 
            'sample_time': np.sum(sample_time_list), 
            'wait_for_thread_time': np.sum(wait_for_thread),
            'eval_time': np.sum(eval_time), 
            'epoch_time_80_percent': np.sum(epoch_time_80_percent)
            }
        
        prefetch_time = {
            'prefetch_compute_time': self.prefetcher.prefetch_compute_time, 
            'eviction_time': self.prefetcher.evict_time,
            'rpc_time': self.prefetcher.rpc_time
        }
        return (np.mean(epoch_time), test_acc, np.mean(forward_time_list), np.mean(backward_time_list), np.mean(update_time_list), 
                np.mean(sample_time_list), np.mean(eval_time),
                self.prefetcher.calculate_hit_rate(), self.prefetcher.calculate_miss_rate(), self.prefetcher.alpha, 
                self.prefetcher.period, self.prefetcher.threshold, absolute_total_time, prefetch_time)
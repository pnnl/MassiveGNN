import numpy as np
import torch as th
import time
from concurrent.futures import ThreadPoolExecutor
import numba
from .lookup import lookup

class Prefetch:
    """
    This is the hybrid version of the prefetcher.
    It uses numba only for lookup.
    Not memory efficient as normal_score is O(n) space.
    """
    def __init__(self, graph, fraction, eviction_period, alpha, halo_nodes, num_layers, train_nid, num_numba_threads, eviction_flag, device, hit_rate_flag):
        print(f"Initializing Prefetch with fraction {fraction} and eviction period {eviction_period}")
        self.graph = graph
        self.fraction = fraction
        self.buffer_length = 0
        self.num_layers = num_layers
        self.train_nid = train_nid
        self.halo_nodes_rank = np.array(list(halo_nodes))
        self.sort_halo_nodes()
        self.prefetch_ids = np.zeros(self.buffer_length, dtype=np.int32)
        self.prefetch_features = th.zeros(self.buffer_length, self.graph.ndata["features"].shape[1])
        self.eviction_score = None  # O(len(buffer)) space initialized in bulk_prefetch
        self.normal_score = np.zeros(self.graph.number_of_nodes(), dtype=np.float32)
        self.alpha = alpha
        self.decay = np.float32(1 - alpha)
        self.period = eviction_period
        self.threshold = round(self.calculate_threshold(), 3)
        self.counter = 0
        self.fetched_features = th.sparse_coo_tensor(
            (self.graph.number_of_nodes(), self.graph.ndata["features"].shape[1]), dtype=th.float32)
        self.rpc_time = 0
        self.prefetch_compute_time = 0
        self.lookup_time = 0
        self.evict_time = 0
        self.update_score_time = 0
        self.prefetch_indices_map = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        # flag to inidicate if prefetch_id is sorted
        self.sorted = False
        self.evict = False
        self.hit = 0
        self.miss = 0
        self.num_numba_threads = num_numba_threads
        self.device = device
        self.hit_rate_flag = hit_rate_flag

    def sort_halo_nodes(self):
        # Sort the halo nodes rank
        self.halo_nodes_rank = np.sort(self.halo_nodes_rank)

    def random_prefetch(self):
        # Randomly select a subset of halo nodes to prefetch
        selected_nodes = np.random.choice(self.halo_nodes_rank, int(len(self.halo_nodes_rank) * self.fraction),
                                          replace=False)
        self.buffer_length = len(selected_nodes)
        self.bulk_prefetch(selected_nodes)

    def degree_based_prefetch(self):
        halo_nodes_tensor = th.tensor(self.halo_nodes_rank)
        # Get the top fraction of nodes by degree
        self.buffer_length = int(len(self.halo_nodes_rank) * self.fraction)
        _, top_indices = th.topk(self.graph.in_degrees(halo_nodes_tensor), self.buffer_length)
        self.bulk_prefetch(halo_nodes_tensor[top_indices])

    def bulk_prefetch(self, nodes):
        # if nodes in numpy array, copy directly
        if isinstance(nodes, np.ndarray):
            self.prefetch_ids = nodes
        else:
            self.prefetch_ids = nodes.numpy()
        self.prefetch_features = self.graph.ndata["features"][nodes]
        self.eviction_score = np.ones(self.buffer_length, dtype=np.float32)
        self.tag_prefetched_nodes_in_normal_score()

    def tag_prefetched_nodes_in_normal_score(self):
        self.normal_score[self.prefetch_ids] = -1

    def sort_prefetch(self):
        sort_idx = np.argsort(self.prefetch_ids)
        self.prefetch_ids = self.prefetch_ids[sort_idx]
        self.prefetch_features = self.prefetch_features[sort_idx]
        self.eviction_score = self.eviction_score[sort_idx]
        # turn on the sorted flag
        self.sorted = True

    def calculate_threshold(self):
        # calculate the threshold for eviction
        return 1 * (1 - self.alpha) ** self.period

    def calculate_hit_rate(self):
        total = self.hit + self.miss
        if total == 0:
            return 0  
        return round(self.hit / total * 100)

    def calculate_miss_rate(self):
        total = self.hit + self.miss
        if total == 0:
            return 0 
        return round(self.miss / total * 100)

    def update_score(self, missed_minibatch_nodes):
        update_score_start = time.time()
        mask = np.nonzero(np.in1d(missed_minibatch_nodes, self.halo_nodes_rank, kind='table'))[0]
        self.normal_score[missed_minibatch_nodes[mask]] += 1
        update_score_end = time.time()
        return update_score_end - update_score_start

    def prefetch(self, input_nodes_array, batch_inputs):
        start_prefetch_compute = time.time()
        self.counter += 1

        # Sort the prefetch_ids
        sort_start = time.time()
        if self.sorted is False:
            self.sort_prefetch()
        sort_end = time.time()

        # Set number of threads for numba
        numba.set_num_threads(self.num_numba_threads)
        lookup_start = time.time()
        # Lookup in the prefetch buffer
        hit_indices, missed_minibatch_idx, feature_indices, self.eviction_score = lookup(input_nodes_array,
                                                                                         self.prefetch_ids,
                                                                                         self.buffer_length,
                                                                                         self.eviction_score,
                                                                                         self.decay)
        lookup_end = time.time()

        copy_features_start = time.time()
        # Copy the features from the prefetch buffer
        batch_inputs[hit_indices] = self.prefetch_features[feature_indices]
        copy_features_end = time.time()

        start_rpc = time.time()
        # RPC for missed minibatch nodes (halo + local nodes)
        batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
        end_rpc = time.time()

        if self.hit_rate_flag:
            count_hit_miss_start = time.time()
            self.hit += len(hit_indices)
            self.miss += np.count_nonzero(
                np.in1d(input_nodes_array[missed_minibatch_idx], self.halo_nodes_rank, kind='table'))
            count_hit_miss_end = time.time()

        end_prefetch_compute = time.time()
        total_rpc_time = (end_rpc - start_rpc)
        self.prefetch_compute_time += (end_prefetch_compute - start_prefetch_compute) - total_rpc_time
        self.rpc_time += total_rpc_time
        total_time = end_prefetch_compute - start_prefetch_compute
        # print all times
        # print(f"""Size of input_nodes_array: {len(input_nodes_array)},
        #       | Size of prefetch_ids: {len(self.prefetch_ids)},
        #       | Total prefetch + RPC time: {end_prefetch_compute - start_prefetch_compute},
        #       | Sort time: {sort_end - sort_start},
        #       | Lookup time: {lookup_end - lookup_start},
        #       | Copy features time: {copy_features_end - copy_features_start},
        #       | Start RPC time: {start_rpc - start_rpc}, | End RPC time: {end_rpc - start_rpc},
        #       | Count hit miss start time: {count_hit_miss_end - count_hit_miss_start},
        #       | Total RPC time: {total_rpc_time}""")
        return batch_inputs

    def prefetch_with_eviction(self, input_nodes_array, batch_inputs):
        start_prefetch_compute = time.time()
        self.counter += 1
        if self.counter % self.period == 0:
            self.evict = True
        else:
            self.evict = False

        # create a mapping from prefetch_ids to prefetch_features indices
        sort_start = time.time()
        if self.sorted is False:
            self.sort_prefetch()
        sort_end = time.time()

        numba.set_num_threads(self.num_numba_threads)
        lookup_start = time.time()
        hit_indices, missed_minibatch_idx, hit_in_buffer, self.eviction_score = lookup(input_nodes_array,
                                                                                       self.prefetch_ids,
                                                                                       self.buffer_length,
                                                                                       self.eviction_score, self.decay)
        lookup_end = time.time()

        copy_features_start = time.time()
        batch_inputs[hit_indices] = self.prefetch_features[hit_in_buffer]
        copy_features_end = time.time()

        if self.hit_rate_flag:
            count_hit_miss_start = time.time()
            self.hit += len(hit_indices)
            self.miss += np.count_nonzero(
                np.in1d(input_nodes_array[missed_minibatch_idx], self.halo_nodes_rank, kind='table'))
            count_hit_miss_end = time.time()

        self.prefetch_compute_time += time.time() - start_prefetch_compute
        total_rpc = evict_start = evict_end = merge_rpc_start = merge_rpc_end = start_evict_rpc = end_evict_rpc = start_none_evict_rpc = end_none_evict_rpc = not_used_in_buffer_start = not_used_in_buffer_end = 0
        
        future = None
        if self.evict:
            evict_start = time.time()
            eviction_candidates_idx, replace_candidates, final_slots = self.replace_eviction_candidates()
            evict_end = time.time()
            if eviction_candidates_idx is not None:
                merge_rpc_start = time.time()
                # Convert to tensor and concatenate
                indices_to_update = th.cat(
                    (th.from_numpy(replace_candidates), th.from_numpy(input_nodes_array[missed_minibatch_idx])))
                # Fetch features
                start_evict_rpc = time.time()
                self.fetched_features = self.rpc(indices_to_update)
                end_evict_rpc = time.time()
                total_rpc += end_evict_rpc - start_evict_rpc
                # Split fetched features back into two groups
                self.prefetch_features[eviction_candidates_idx] = self.fetched_features[:final_slots]
                batch_inputs[missed_minibatch_idx] = self.fetched_features[final_slots:]
                # turn off the sorted flag
                self.sorted = False
                merge_rpc_end = time.time()
            else:
                future = self.executor.submit(self.update_score, input_nodes_array[missed_minibatch_idx])
                no_evict_rpc_start = time.time()  # when no eviction candidates
                batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
                total_rpc += time.time() - no_evict_rpc_start
        else:
            future = self.executor.submit(self.update_score, input_nodes_array[missed_minibatch_idx])
            start_normal_rpc = time.time()
            batch_inputs[missed_minibatch_idx] = self.rpc(input_nodes_array[missed_minibatch_idx])
            total_rpc += time.time() - start_normal_rpc

        # wait for update_score_thread to finish
        if future is not None:
            wait_start = time.time()
            update_score_time = future.result()
            self.update_score_time += update_score_time
            wait_end = time.time()
            wait_time = wait_end - wait_start
        else:
            wait_time = 0
            update_score_time = 0

        eviction_time = (evict_end - evict_start)
        self.rpc_time += total_rpc
        self.evict_time += eviction_time + (merge_rpc_end - merge_rpc_start) - (end_evict_rpc - start_evict_rpc)

        # print all times
        # print(f"""Size of input_nodes_array: {len(input_nodes_array)},
        #       | Size of prefetch_ids: {len(self.prefetch_ids)},
        #       | Sort time: {sort_end - sort_start},
        #       | Lookup time: {lookup_end - lookup_start},
        #       | Copy features time: {copy_features_end - copy_features_start},
        #       | Evict and find replacement time: {eviction_time},
        #       | Merge RPC time: {merge_rpc_end - merge_rpc_start},
        #       | Eviction RPC time: {end_evict_rpc - start_evict_rpc},
        #       | Total RPC time: {total_rpc},
        #       | Update score run time: {update_score_time},
        #       | Update score wait time: {wait_time},
        #       | Total time: {time.time() - start_prefetch_compute}""")

        return batch_inputs

    def rpc(self, node_idx):
        features = self.graph.ndata["features"][node_idx]
        return features

    def find_eviction_candidates(self, desired_slots=None):
        # select the nodes with eviction score < threshold and return their indices and how many of them are there
        below_threshold_mask = self.eviction_score < self.threshold
        eviction_candidates_idx = np.nonzero(below_threshold_mask)[0]
        slots = np.count_nonzero(below_threshold_mask)

        if slots == 0:
            return None, None, 0  # No eviction candidates
        # If a specific desired_slots is given and is less than the current slots, use it instead.
        if desired_slots is not None and desired_slots < slots:
            slots = desired_slots
            eviction_candidates_idx = eviction_candidates_idx[:slots]  # TODO:does this need to be sorted?

        return eviction_candidates_idx, self.eviction_score[eviction_candidates_idx], slots

    def find_replace_candidates(self):
        # Step 1: Get the scores of the halo nodes.
        halo_scores = self.normal_score[self.halo_nodes_rank]

        # Step 2: Sort these scores in descending order and get their indices.
        sorted_indices_within_halo = np.argsort(-halo_scores)

        # Filter out the indices corresponding to scores of 0.
        valid_indices = sorted_indices_within_halo[halo_scores[sorted_indices_within_halo] > 0]

        # Map these indices back to the original self.normal_score array.
        replace_candidates = self.halo_nodes_rank[valid_indices]
        return replace_candidates, self.normal_score[replace_candidates]

    def replace_eviction_candidates(self):
        eviction_candidates_idx, eviction_score, max_eviction_slots = self.find_eviction_candidates()

        # If no eviction candidates, exit early.
        if max_eviction_slots == 0:
            return None, None, 0

        replace_candidates, normal_score = self.find_replace_candidates()
        final_slots = min(max_eviction_slots, len(replace_candidates))
        
        # truncate to final_slots
        replace_candidates = replace_candidates[:final_slots]
        eviction_candidates_idx = eviction_candidates_idx[:final_slots]
        eviction_candidates = self.prefetch_ids[eviction_candidates_idx]
        eviction_score = eviction_score[:final_slots]
        normal_score = normal_score[:final_slots]

        self.prefetch_ids[eviction_candidates_idx] = replace_candidates
        self.normal_score[eviction_candidates] = eviction_score
        self.tag_prefetched_nodes_in_normal_score()

        # copy normal scores of the replaced nodes to the eviction scores as they are the new prefetch_ids
        self.eviction_score[eviction_candidates_idx] = normal_score
        return eviction_candidates_idx, replace_candidates, final_slots

    def close(self):
        self.executor.shutdown()
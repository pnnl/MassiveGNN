import torch as th
import numpy as np
import dgl

def get_halo_in_hops(g, pb, train_nid, num_hops, debug=False):
    
    remote_lnid = th.nonzero(g.local_partition.ndata["inner_node"] == False).squeeze()
    print(f"part {g.rank()} Number of remote nodes: {len(remote_lnid)}")

    if debug:
        print("Debug mode: returning all remote nodes")
        return g.local_partition.ndata[dgl.NID][remote_lnid].detach().numpy()
    src, dst = g.local_partition.edges()
    
    # convert train_nid to local node ids
    train_lnid = pb.nid2localnid(train_nid, pb.partid).detach().numpy()
    
    hop_halo_nodes_set = set()
    current_nodes_set = set(train_lnid.tolist())
    visited_nodes_set = set()

    print(f"Part {g.rank()}: Finding halo nodes in {num_hops} hops")
    for _ in range(num_hops):
        next_nodes_set = set()
        
        # Find indices where dst nodes are in the current_nodes_set
        current_dst_indices = np.where(np.isin(dst, list(current_nodes_set)))[0]
        current_src_nodes = src[current_dst_indices]
        
        # Exclude nodes that point back to already visited nodes
        mask = ~np.isin(current_src_nodes, visited_nodes_set)
        current_src_nodes = current_src_nodes[mask]

        for node in current_src_nodes:
            if node.item() in remote_lnid:
                hop_halo_nodes_set.add(node.item())
            else:
                next_nodes_set.add(node.item())
        # print(f"Number of unique halo nodes after {h} hops: {len(hop_halo_nodes_set)}")
        visited_nodes_set.update(current_nodes_set)
        current_nodes_set = next_nodes_set

    print(f"Number of unique halo nodes after {num_hops} hops: {len(hop_halo_nodes_set)}")

    return g.local_partition.ndata[dgl.NID][list(hop_halo_nodes_set)].detach().numpy()


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

def compute_acc(pred, labels):
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

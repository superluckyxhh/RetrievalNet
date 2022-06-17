import torch
import torch.nn as nn
import core.distributed as dist


def reduce_dict(input_dict, average=True):
    world_size = dist.get_world_size()
    
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if average:
            values /= world_size

        reduce_dict = {k: v for k, v in zip(names, values)}
        
    return reduce_dict


def scaled_all_reduce(tensors):
    world_size = dist.get_world_size()
    
    if world_size < 2:
        return tensors
    
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)

    for reduction in reductions:
        reduction.wait()

    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / world_size)
        
    return tensors
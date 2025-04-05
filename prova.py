import torch
import torch.distributed as dist

dist.init_process_group("gloo", rank=0, world_size=1)
print("Distributed initialized!")


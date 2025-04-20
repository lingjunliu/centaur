import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

# torch.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, out=None)
def torch_version(input_dict, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack inputs from dictionary
    LU_data = torch.tensor(input_dict["LU_data"])
    LU_pivots = torch.tensor(input_dict["LU_pivots"])
    unpack_data = input_dict.get("unpack_data", True)
    unpack_pivots = input_dict.get("unpack_pivots", True)
    
    if not cpu:
        LU_data = LU_data.cuda()
        LU_pivots = LU_pivots.cuda()
    
    # Perform torch addition
    P, L, U = torch.lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots)
    
    # Move result to CPU for consistent return format
    if not cpu:
        if P is not None:
            P = P.cpu()
        if L is not None:
            L = L.cpu()
        if U is not None:
            U = U.cpu()
    
    return { "P": P.numpy() if P is not None else None, "L": L.numpy() if L is not None else None, "U": U.numpy() if U is not None else None }

# TODO: Implement the TensorFlow version of the lu_unpack driver function
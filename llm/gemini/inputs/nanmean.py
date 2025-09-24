
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def nanmean_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D tensor with NaNs, dim=0
    input_tensor = torch.tensor([[float('nan'), 1.0, 2.0], [3.0, float('nan'), 5.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 1D tensor with NaNs, no dim specified
    input_tensor = torch.tensor([float('nan'), 1.0, 2.0, float('nan')]).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensor with NaNs, dim=(0, 2), keepdim=True
    input_tensor = torch.randn(2, 3, 4).float()
    input_tensor[0, 1, 2] = float('nan')
    input_tensor[1, 0, 0] = float('nan')
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor, "dim": (0, 2), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Tensor with all NaNs in a dimension
    input_tensor = torch.tensor([[float('nan'), float('nan')], [1.0, 2.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Negative values and NaNs
    input_tensor = torch.tensor([[-1.0, float('nan'), -2.0], [float('nan'), -3.0, 4.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": (1,), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: 4D tensor
    input_tensor = torch.randn(2, 2, 2, 2).float()
    input_tensor[0, 0, 0, 0] = float('nan')
    input_tensor[1, 1, 1, 1] = float('nan')
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor, "dim": (0, 2), "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = nanmean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nanmean', generated_inputs)

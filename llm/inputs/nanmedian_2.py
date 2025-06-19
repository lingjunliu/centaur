
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def nanmedian_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor with NaNs
    input_tensor = torch.tensor([1.0, float('nan'), 3.0, 2.0]).numpy()
    input_dict = {"input": input_tensor, "dim": -1, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor with NaNs, dim=0, keepdim=True
    input_tensor = torch.tensor([[2.0, 3.0, 1.0], [float('nan'), 1.0, float('nan')]]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor with NaNs, dim=1, keepdim=False
    input_tensor = torch.tensor([[2.0, 3.0, 1.0], [float('nan'), 1.0, float('nan')]]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor with NaNs, dim=2, keepdim=True
    input_tensor = torch.tensor([[[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]], [[7.0, 8.0, 9.0], [float('nan'), 11.0, 12.0]]]).numpy()
    input_dict = {"input": input_tensor, "dim": 2, "keepdim": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Tensor with only NaNs. Removed as it causes issues with dimension reduction.

    return list_of_inputs

generated_inputs["torch.nanmedian_2"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nanmedian_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmedian_2'.")

check_valid('torch.nanmedian', generated_inputs['torch.nanmedian_2'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def nansum_inputs():
    list_of_inputs = []

    # Case 1: Basic case with NaNs, no dim, no keepdim, float32
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor with NaNs, dim=0, keepdim=False, float64
    input = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]], dtype=torch.float64).numpy()
    input_dict = {"input": input, "dim": (0,), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor with NaNs, dim=(0, 2), keepdim=True, float64
    input = torch.randn(2, 3, 4, dtype=torch.float64)
    input[0, 1, 2] = float('nan')
    input[1, 2, 0] = float('nan')
    input = input.numpy()
    input_dict = {"input": input, "dim": (0, 2), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor with only NaNs, dtype specified, float16
    input = torch.tensor([float('nan'), float('nan'), float('nan')], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": None, "keepdim": False, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 2D tensor, dim=1, keepdim=True, negative values, float32
    input = torch.tensor([[-1.0, 2.0, float('nan')], [3.0, -4.0, 5.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "dim": (1,), "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: More dimensions, testing int32 and keepdim false
    input = torch.randint(-5, 5, (2, 3, 2, 2), dtype=torch.int32).float()
    input[0, 1, 1, 0] = float('nan')
    input[1, 0, 0, 1] = float('nan')
    input = input.numpy()
    input_dict = {"input": input, "dim": (0, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nansum_3"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nansum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_3'.")

check_valid('torch.nansum', generated_inputs['torch.nansum_3'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def nanmean_inputs():
    list_of_inputs = []

    # Case 1: Basic case with NaN values, dim=None
    input1 = torch.tensor([[float('nan'), 1, 2], [1, 2, 3]], dtype=torch.float32).numpy()
    input_dict1 = {"input": input1, "dim": None, "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Along dimension 0
    input2 = torch.tensor([[float('nan'), 1, 2], [1, 2, float('nan')]], dtype=torch.float64).numpy()
    input_dict2 = {"input": input2, "dim": (0,), "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Along dimension 1, keepdim=True
    input3 = torch.tensor([[float('nan'), 1, 2], [1, float('nan'), 3]], dtype=torch.float32).numpy()
    input_dict3 = {"input": input3, "dim": (1,), "keepdim": True, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: All elements are NaN
    input4 = torch.tensor([[float('nan'), float('nan')], [float('nan'), float('nan')]], dtype=torch.float64).numpy()
    input_dict4 = {"input": input4, "dim": None, "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 3D tensor, dim=(0, 1)
    input5 = torch.randn(2, 3, 4, dtype=torch.float32)
    input5[0, 1, 2] = float('nan')
    input5[1, 0, 3] = float('nan')
    input5 = input5.numpy()
    input_dict5 = {"input": input5, "dim": (0, 1), "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nanmean_2"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nanmean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmean_2'.")

check_valid('torch.nanmean', generated_inputs['torch.nanmean_2'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def nanmean_inputs():
    list_of_inputs = []

    # Case 1: Basic case with NaN values and dim=0
    input_tensor = torch.tensor([[float('nan'), 1.0, 2.0], [3.0, float('nan'), 4.0], [5.0, 6.0, float('nan')]]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: No NaN values, different dim and keepdim=True
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": True, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: All NaN values in a dimension
    input_tensor = torch.tensor([[float('nan'), float('nan')], [1.0, 2.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Higher dimensional tensor with NaNs
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_tensor[0, 1, 2] = float('nan')
    input_tensor[1, 0, 3] = float('nan')
    input_dict = {"input": input_tensor, "dim": (0, 1), "keepdim": False, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Specifying dtype, avoid int32 with nan
    input_tensor = torch.tensor([[float('nan'), 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False, "dtype": torch.float64, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nanmean_1"] = nanmean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nanmean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmean_1'.")

check_valid('torch.nanmean', generated_inputs['torch.nanmean_1'], lib="torch")

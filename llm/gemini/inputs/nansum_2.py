
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor with NaNs
    input1 = torch.tensor([1., 2., float('nan'), 4.]).numpy()
    input_dict1 = {"input": input1, "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D tensor with NaNs, sum along dim=0
    input2 = torch.tensor([[1., 2.], [3., float('nan')]]).numpy()
    input_dict2 = {"input": input2, "dim": 0, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 2D tensor with NaNs, sum along dim=1, keepdim=True
    input3 = torch.tensor([[1, 2], [3., float('nan')]]).numpy()
    input_dict3 = {"input": input3, "dim": 1, "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 3D tensor with NaNs, sum along multiple dimensions
    input4 = torch.randn(2, 3, 4)
    input4[0, 1, 2] = float('nan')
    input4[1, 0, 3] = float('nan')
    input4 = input4.numpy()
    input_dict4 = {"input": input4, "dim": (0, 2), "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Tensor with dtype specified
    input5 = torch.tensor([1, 2, -3, float('nan')]).numpy()
    input_dict5 = {"input": input5, "dim": None, "keepdim": False, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_2'.")

check_valid('torch.nansum', generated_inputs['torch.nansum_2'], lib="torch")

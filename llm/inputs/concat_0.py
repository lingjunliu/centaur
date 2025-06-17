
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def concat_inputs():
    list_of_inputs = []

    # Case 1: Basic concatenation along dim=0 (default)
    tensors = [torch.randn(2, 3).numpy(), torch.randn(3, 3).numpy()]
    input_dict = {"tensors": tensors, "dim": 0, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Concatenation along dim=1
    tensors = [torch.randn(2, 3).numpy(), torch.randn(2, 4).numpy()]
    input_dict = {"tensors": tensors, "dim": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Concatenation with different data types (int and float) - Removed due to errors and type constraint issue
    # tensors = [torch.randint(0, 10, (2, 3)).numpy().astype(np.float32), torch.randn(2, 3).numpy()]
    # input_dict = {"tensors": tensors, "dim": 0, "out": None}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Concatenation with 3D tensors
    tensors = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy()]
    input_dict = {"tensors": tensors, "dim": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Concatenation with negative dim
    tensors = [torch.randn(2, 3).numpy(), torch.randn(2, 3).numpy()]  # Fix: Sizes must match
    input_dict = {"tensors": tensors, "dim": -1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.concat"] = concat_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.concat'.")

check_valid('torch.concat', generated_inputs['torch.concat'], lib="torch")

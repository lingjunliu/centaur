
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def repeat_interleave_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor, scalar repeats
    input_tensor = torch.tensor([1, 2, 3])
    repeats = 2
    dim = 0
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D tensor, scalar repeats
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = 3
    dim = 0
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor, tensor repeats
    input_tensor = torch.tensor([1, 2, 3])
    repeats = torch.tensor([1, 2, 3])
    dim = 0
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D tensor, tensor repeats, dim=1
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = torch.tensor([2, 1])
    dim = 1
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats.numpy(), "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: 3D tensor, scalar repeats
    input_tensor = torch.randn(2, 3, 4)
    repeats = 2
    dim = 1
    input_dict = {"input": input_tensor.numpy(), "repeats": repeats, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.repeat_interleave_1"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.repeat_interleave_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_1'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_1'], lib="torch")

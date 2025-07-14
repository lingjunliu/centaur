
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def rsub_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    other_tensor = torch.tensor([4.0, 5.0, 6.0])
    alpha_val = 1.0

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    alpha_val = 0.5

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    other_tensor = torch.tensor([4, 5, 6], dtype=torch.int32)
    alpha_val = 2.0

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.tensor([1.0])
    other_tensor = torch.tensor([5.0])
    alpha_val = 1.0

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    other_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    alpha_val = 1.0

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.tensor([1.5, 2.5, 3.5])
    other_tensor = torch.tensor([4.5, 5.5, 6.5])
    alpha_val = 0.0

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    other_tensor = torch.tensor([4, 5, 6], dtype=torch.int64)
    alpha_val = 1.5

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32)
    other_tensor = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=torch.int32)
    alpha_val = 1.0
    
    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    other_tensor = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    alpha_val = 0.5

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.tensor([1, 2], dtype=torch.int32)
    other_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    alpha_val = 1

    input_dict = {
        "input": input_tensor.numpy(),
        "other": other_tensor.numpy(),
        "alpha": alpha_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rsub"] = rsub_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rsub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsub'.")

check_valid('torch.rsub', generated_inputs['torch.rsub'], lib="torch", suffix=0)

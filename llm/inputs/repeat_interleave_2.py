
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def repeat_interleave_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.tensor([1, 2, 3])
    repeats_tensor = torch.tensor(2)
    dim_val = 0

    input = input_tensor.numpy()
    repeats = repeats_tensor.numpy()
    dim = dim_val

    input_dict = {
        "input": input,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats_tensor = torch.tensor([1, 2])
    dim_val = 0
    
    input = input_tensor.numpy()
    repeats = repeats_tensor.numpy()
    dim = dim_val

    input_dict = {
        "input": input,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats_tensor = torch.tensor([2, 1])
    dim_val = 1

    input = input_tensor.numpy()
    repeats = repeats_tensor.numpy()
    dim = dim_val

    input_dict = {
        "input": input,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
    repeats_tensor = torch.tensor(2)
    dim_val = 0

    input = input_tensor.numpy()
    repeats = repeats_tensor.numpy()
    dim = dim_val
    input_dict = {
        "input": input,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
    repeats_tensor = torch.tensor([1,2])
    dim_val = 1

    input = input_tensor.numpy()
    repeats = repeats_tensor.numpy()
    dim = dim_val

    input_dict = {
        "input": input,
        "repeats": repeats,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.repeat_interleave_2"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.repeat_interleave_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_2'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_2'], lib="torch", suffix=2)

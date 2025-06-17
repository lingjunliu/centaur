
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_narrow_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor narrowing
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 0,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Narrowing along a different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "start": 1,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Using a negative start index
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": -1,
        "start": -1,
        "length": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensor narrowing
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "start": 0,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Narrowing to a single element
    input_tensor = torch.randn(5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 2,
        "length": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.narrow"] = torch_narrow_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.narrow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.narrow'.")

check_valid('torch.narrow', generated_inputs['torch.narrow'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def topk_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with a 1D tensor
    input_dict = {
        "input": torch.randn(5).numpy(),
        "k": 3,
        "dim": None,
        "largest": True,
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor with largest=False
    input_dict = {
        "input": torch.randn(3, 4).numpy(),
        "k": 2,
        "dim": 1,
        "largest": False,
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 3D tensor with different dim and sorted=False
    input_dict = {
        "input": torch.randn(2, 3, 5).numpy(),
        "k": 2,
        "dim": 2,
        "largest": True,
        "sorted": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Integer tensor
    input_dict = {
        "input": torch.randint(0, 10, (4, 4)).numpy(),
        "k": 3,
        "dim": 1,
        "largest": True,
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Negative values and k=1
    input_dict = {
        "input": torch.randn(3, 3) * -1.0,
        "k": 1,
        "dim": 1,
        "largest": True,
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.topk"] = topk_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.topk' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.topk'.")

check_valid('torch.topk', generated_inputs['torch.topk'], lib="torch")

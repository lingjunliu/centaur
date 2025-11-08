
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unique_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1, 3, 2, 3], dtype=torch.long).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([-1, -1, 0, 2, 2, 2], dtype=torch.int64).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": False,
        "return_inverse": True,
        "return_counts": True,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1.0, 2.0],
                              [1.0, 2.0],
                              [3.0, 4.0]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[0.1, 0.1, 1.1],
                              [0.2, 0.2, 2.2]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": False,
        "return_inverse": False,
        "return_counts": True,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a0 = torch.tensor([[1, 2],
                       [3, 4]], dtype=torch.uint8)
    a1 = torch.tensor([[1, 2],
                       [3, 4]], dtype=torch.uint8)
    a2 = torch.tensor([[5, 6],
                       [7, 8]], dtype=torch.uint8)
    input_arr = torch.stack([a0, a1, a2], dim=2).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    s0 = torch.tensor([[1, 1],
                       [2, 2]], dtype=torch.int32)
    s1 = torch.tensor([[3, 3],
                       [4, 4]], dtype=torch.int32)
    s2 = torch.tensor([[1, 1],
                       [2, 2]], dtype=torch.int32)
    input_arr = torch.stack([s0, s1, s2], dim=1).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": False,
        "return_inverse": True,
        "return_counts": True,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[True, True, False, True],
                              [False, False, True, True],
                              [True, True, False, False]], dtype=torch.bool).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": True,
        "return_inverse": True,
        "return_counts": True,
        "dim": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    b0 = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]], dtype=torch.int64)
    b1 = b0.clone()
    b2 = torch.tensor([[[13, 14], [15, 16], [17, 18]],
                       [[19, 20], [21, 22], [23, 24]]], dtype=torch.int64)
    input_arr = torch.stack([b0, b1, b2], dim=2).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": False,
        "return_inverse": False,
        "return_counts": False,
        "dim": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([1.0, 1.0, 2.0, 3.0, 2.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": False,
        "return_inverse": True,
        "return_counts": True,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[1, 1, 2, 2, 3],
                              [4, 4, 5, 5, 6]], dtype=torch.uint8).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    u0 = torch.tensor([[[1.0, 2.0],
                        [3.0, 4.0]],
                       [[5.0, 6.0],
                        [7.0, 8.0]]], dtype=torch.float32)
    u1 = u0.clone()
    u2 = u0 + 10.0
    input_arr = torch.stack([u0, u1, u2], dim=0).numpy()
    input_dict = {
        "input": input_arr,
        "sorted": False,
        "return_inverse": True,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.unique"] = unique_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unique'.")


check_valid('torch.unique', generated_inputs['torch.unique'], lib="torch", suffix=0)

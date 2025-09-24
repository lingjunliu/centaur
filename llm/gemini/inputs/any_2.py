
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def any_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D boolean tensor
    input_dict = {
        "input": np.array([[True, False], [False, True]]),
        "dim": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D integer tensor with dim=0
    input_dict = {
        "input": np.array([[[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0]]]),
        "dim": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float tensor with negative values, dim=0
    input_dict = {
        "input": np.array([-1.0, 0.0, 2.5]),
        "dim": 0,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D complex tensor, dim=(1, 2)
    input_dict = {
        "input": np.array([[[[1+1j, 0+0j], [0+0j, 1+1j]], [[0+0j, 0+0j], [0+0j, 0+0j]]]]),
        "dim": (1, 2),
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean tensor, dim=None, keepdim=False
    input_dict = {
        "input": np.array([[False, False], [False, False]]),
        "dim": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D uint8 tensor
    input_dict = {
        "input": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8),
        "dim": 2,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Tensor with all zeros
    input_dict = {
        "input": np.zeros((2, 3), dtype=np.int32),
        "dim": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.any_2"] = any_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.any_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.any_2'.")

check_valid('torch.any', generated_inputs['torch.any_2'], lib="torch")

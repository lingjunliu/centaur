
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_std_inputs():
    list_of_inputs = []

    # Input 1: Basic case with dim=1, keepdim=True
    input1 = torch.tensor([[0.2035, 1.2959, 1.8101, -0.4644],
                           [1.5027, -0.3270, 0.5905, 0.6538],
                           [-1.5745, 1.3330, -0.5596, -0.6548],
                           [0.1264, -0.5080, 1.6420, 0.1992]]).numpy()
    input_dict1 = {
        "input": input1,
        "dim": (1,),
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different dim, keepdim=False, different correction
    input2 = torch.randn(3, 4, 5).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0, 2),
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single dimension tensor
    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "dim": None,
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values and correction = 2
    input4 = torch.tensor([[-1.0, -2.0, -3.0],
                           [-4.0, -5.0, -6.0]]).numpy()
    input_dict4 = {
        "input": input4,
        "dim": (0,),
        "correction": 2,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor with dim as a tuple
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "dim": (1, 2),
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.std_2"] = torch_std_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_2'.")

check_valid('torch.std', generated_inputs['torch.std_2'], lib="torch")

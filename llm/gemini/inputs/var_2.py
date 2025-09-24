
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_var_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor, dim=1, keepdim=True
    input1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict1 = {
        "input": input1,
        "dim": (1,),
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 3D tensor, dim=(0, 2), keepdim=False
    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0, 2),
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 1D tensor, no dim specified, keepdim=False (implicitly)
    input3 = torch.arange(1, 6, dtype=torch.float32).numpy()
    input_dict3 = {
        "input": input3,
        "dim": None,
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Tensor with negative values, dim=0, keepdim=True
    input4 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]).numpy()
    input_dict4 = {
        "input": input4,
        "dim": (0,),
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Float64 tensor, dim=1, keepdim=False
    input5 = torch.randn(3, 5, dtype=torch.float64).numpy()
    input_dict5 = {
        "input": input5,
        "dim": (1,),
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.var_2"] = torch_var_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_2'.")

check_valid('torch.var', generated_inputs['torch.var_2'], lib="torch")

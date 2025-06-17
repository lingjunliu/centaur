
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "input": input1,
        "dim": (0,),
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor, dim=0
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0,),
        "unbiased": False,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D float tensor, dim=1
    input3 = torch.randn(3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "dim": (1,),
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor, dim=(0, 2)
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {
        "input": input4,
        "dim": (0, 2),
        "unbiased": False,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D int tensor, dim=0
    input5 = torch.randint(0, 10, (3, 4)).numpy()
    input_dict5 = {
        "input": input5,
        "dim": (0,),
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D float tensor without dim specified.
    input6 = torch.randn(5).numpy()
    input_dict6 = {
        "input": input6,
        "unbiased": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.std_mean_2"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_2'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_2'], lib="torch")

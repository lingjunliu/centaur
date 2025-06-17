
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor, dim=0
    input_1 = torch.randn(5).numpy()
    input_dict_1 = {
        "input": input_1,
        "dim": (0,),
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int tensor, dim=1, keepdim=True
    input_2 = torch.randint(-5, 5, (3, 4)).numpy()
    input_dict_2 = {
        "input": input_2,
        "dim": (1,),
        "unbiased": False,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float tensor, dim=(0,2)
    input_3 = torch.randn(2, 3, 4).numpy()
    input_dict_3 = {
        "input": input_3,
        "dim": (0,2),
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: 4D complex tensor, dim=2
    input_4 = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict_4 = {
        "input": input_4,
        "dim": (2,),
        "unbiased": False,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D float tensor, dim=None (all dims)
    input_5 = torch.randn(3, 4).numpy()
    input_dict_5 = {
        "input": input_5,
        "dim": None,
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.std_mean_3"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_3'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_3'], lib="torch")

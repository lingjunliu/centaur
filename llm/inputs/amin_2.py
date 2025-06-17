
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_amin_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, dim=1, keepdim=False
    input1 = torch.randn(4, 4).numpy()
    dim1 = (1,)
    keepdim1 = False
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, dim=(0, 2), keepdim=True
    input2 = torch.randint(-5, 5, (2, 3, 4)).numpy()
    dim2 = (0, 2)
    keepdim2 = True
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.amin_2"] = torch_amin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.amin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.amin_2'.")

check_valid('torch.amin', generated_inputs['torch.amin_2'], lib="torch")

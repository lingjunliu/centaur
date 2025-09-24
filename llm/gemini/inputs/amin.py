
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_amin_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor, dim=1, keepdim=False
    input_tensor = torch.randn(4, 4).numpy()
    dim_value = 1
    keepdim_value = False
    input_dict = {"input": input_tensor, "dim": dim_value, "keepdim": keepdim_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D int tensor, dim=0, keepdim=True
    input_tensor = torch.randint(-5, 5, (2, 3, 4)).numpy()
    dim_value = 0
    keepdim_value = True
    input_dict = {"input": input_tensor, "dim": dim_value, "keepdim": keepdim_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D float tensor with negative values, dim=(1, 2), keepdim=False
    input_tensor = (torch.randn(2, 3, 4, 5) * -1).numpy()
    dim_value = (1, 2)
    keepdim_value = False
    input_dict = {"input": input_tensor, "dim": dim_value, "keepdim": keepdim_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float tensor, dim=0, keepdim=False
    input_tensor = torch.randn(5).numpy()
    dim_value = 0
    keepdim_value = False
    input_dict = {"input": input_tensor, "dim": dim_value, "keepdim": keepdim_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 5D float tensor, dim=(0, 2, 4), keepdim=True
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    dim_value = (0, 2, 4)
    keepdim_value = True
    input_dict = {"input": input_tensor, "dim": dim_value, "keepdim": keepdim_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int tensor, dim=0, keepdim=False
    input_tensor = torch.randint(0, 10, (5, 3)).numpy()
    dim_value = 0
    keepdim_value = False
    input_dict = {"input": input_tensor, "dim": dim_value, "keepdim": keepdim_value}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = torch_amin_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('amin', generated_inputs)

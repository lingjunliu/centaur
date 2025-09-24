
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def native_channel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic 4D tensor, groups = 2
    input1 = torch.randn(1, 4, 10, 10).numpy()
    groups1 = 2
    input_dict1 = {"input": input1, "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4D tensor, groups = 4 (equal to number of channels)
    input2 = torch.randn(1, 4, 5, 5).numpy()
    groups2 = 4
    input_dict2 = {"input": input2, "groups": groups2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor, groups = 1 (no shuffle)
    input3 = torch.randn(1, 8, 7, 7).numpy()
    groups3 = 1
    input_dict3 = {"input": input3, "groups": groups3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 4D tensor, different batch size
    input4 = torch.randn(2, 6, 8, 8).numpy()
    groups4 = 3
    input_dict4 = {"input": input4, "groups": groups4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different spatial dimensions
    input5 = torch.randn(1, 12, 15, 20).numpy()
    groups5 = 4
    input_dict5 = {"input": input5, "groups": groups5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6:  3D tensor
    input6 = torch.randn(3, 6, 6).numpy()
    groups6 = 2
    input_dict6 = {"input": input6, "groups": groups6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float64 tensor
    input7 = torch.randn(1, 4, 10, 10, dtype=torch.float64).numpy()
    groups7 = 2
    input_dict7 = {"input": input7, "groups": groups7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
        
    return list_of_inputs

generated_inputs = native_channel_shuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('native_channel_shuffle', generated_inputs)


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ceil__inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float tensor
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with negative values
    input2 = (torch.randn(3, 4) * 5 - 2).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float tensor
    input4 = torch.randn(1).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large float tensor
    input5 = torch.randn(10, 10, 10).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: float tensor with some NaN and Inf values
    input6 = torch.randn(3, 3).numpy()
    input6[0, 0] = np.nan
    input6[1, 1] = np.inf
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: tensor with a zero
    input7 = torch.zeros(2,2).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.ceil_"] = ceil__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ceil_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ceil_'.")

check_valid('torch.ceil_', generated_inputs['torch.ceil_'], lib="torch")

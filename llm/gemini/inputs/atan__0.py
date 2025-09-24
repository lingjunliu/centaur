
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def atan__inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = torch.randn(2, 2) * -1.0
    input2 = input2.numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor
    input3 = torch.randn(5).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large values
    input5 = (torch.randn(2, 2) * 100).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Zero values
    input6 = torch.zeros(3, 3).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Scalar tensor
    input7 = torch.tensor(3.14).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.atan_"] = atan__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atan_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atan_'.")

check_valid('torch.atan_', generated_inputs['torch.atan_'], lib="torch")

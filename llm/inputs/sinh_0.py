
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def sinh_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input1 = torch.randn(4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor (converted to float)
    input2 = torch.randint(-5, 5, (3, 3)).float().numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values
    input3 = (torch.randn(2, 2) * -1.0).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multi-dimensional tensor
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Zero tensor
    input5 = torch.zeros(5).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs["torch.sinh"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sinh'.")

check_valid('torch.sinh', generated_inputs['torch.sinh'], lib="torch")

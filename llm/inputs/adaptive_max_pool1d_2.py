
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(1, 3, 6).numpy()
    output_size1 = (2,)
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor (converted to float)
    input2 = torch.randint(0, 10, (2, 4, 8)).float().numpy()
    output_size2 = (3,)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values
    input3 = torch.randn(1, 2, 5) * -1.0
    input3 = input3.numpy()
    output_size3 = (1,)
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different output size
    input4 = torch.randn(2, 5, 10).numpy()
    output_size4 = (5,)
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger input tensor
    input5 = torch.randn(4, 16, 32).numpy()
    output_size5 = (8,)
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool1d_2"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_max_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool1d_2'.")

check_valid('torch.nn.functional.adaptive_max_pool1d', generated_inputs['torch.nn.functional.adaptive_max_pool1d_2'], lib="torch")

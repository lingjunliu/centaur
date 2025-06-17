
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor and a tuple output size
    input1 = torch.randn(1, 3, 32, 32).numpy()
    output_size1 = (8, 8)
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor with a single integer output size (square output)
    input2 = torch.randint(0, 10, (2, 4, 16, 16)).float().numpy()
    output_size2 = (7, 7)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 4: Negative values in the input tensor
    input4 = torch.randn(1, 1, 20, 20) * -1.0
    input4 = input4.numpy()
    output_size4 = (5, 5)
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Different input dimensions, smaller input size, non-square output
    input5 = torch.randn(2, 1, 10, 12).numpy()
    output_size5 = (2, 3)
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool2d_1"] = adaptive_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.adaptive_max_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool2d_1'.")

check_valid('torch.nn.functional.adaptive_max_pool2d', generated_inputs['torch.nn.functional.adaptive_max_pool2d_1'], lib="torch")

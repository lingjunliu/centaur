
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def ConstantPad1d_inputs():
    list_of_inputs = []

    # Test case 1: Single padding value, float input
    input = torch.randn(1, 2, 4).numpy()
    padding = 2
    value = 3.5
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Tuple padding, int input
    input = torch.randint(0, 10, (1, 3, 5)).numpy()
    padding = (1, 2)
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Different shaped input with negative padding and value
    input = torch.randn(2, 5, 3).numpy()
    padding = (0, 3)
    value = -1.0
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Input with only one dimension
    input = torch.randn(1, 1, 7).numpy()
    padding = 1
    value = 2.0
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Larger padding values
    input = torch.randn(1, 4, 2).numpy()
    padding = (4, 5)
    value = 1.5
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Zero padding on one side
    input = torch.randn(1, 2, 5).numpy()
    padding = (0, 2)
    value = 4.0
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 7: Different input shape again
    input = torch.randn(3, 1, 6).numpy()
    padding = (2, 1)
    value = -2.5
    input_dict = {"padding": padding, "value": value, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConstantPad1d_2"] = ConstantPad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConstantPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad1d_2'.")

check_valid('torch.nn.ConstantPad1d', generated_inputs['torch.nn.ConstantPad1d_2'], lib="torch")

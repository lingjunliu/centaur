
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ZeroPad2d_inputs():
    list_of_inputs = []

    # Test case 1: int padding
    input = torch.randn(1, 1, 3, 3).numpy()
    padding = 2
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: tuple padding
    input = torch.randn(1, 1, 3, 3).numpy()
    padding = (1, 1, 2, 0)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Different input shape (C, H, W)
    input = torch.randn(3, 5, 5).numpy()
    padding = 1
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 4: Different input shape (N, C, H, W) and tuple padding
    input = torch.randn(2, 3, 4, 4).numpy()
    padding = (0, 2, 1, 3)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Larger padding values
    input = torch.randn(1, 1, 2, 2).numpy()
    padding = 5
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Non-square input
    input = torch.randn(1, 1, 4, 6).numpy()
    padding = 2
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 7: Non-unit channel
    input = torch.randn(1, 3, 4, 4).numpy()
    padding = (1, 0, 2, 0)
    input_dict = {"input": input, "padding": padding}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ZeroPad2d_1"] = ZeroPad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ZeroPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad2d_1'.")

check_valid('torch.nn.ZeroPad2d', generated_inputs['torch.nn.ZeroPad2d_1'], lib="torch")

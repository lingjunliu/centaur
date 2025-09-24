
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReflectionPad2d_inputs():
    list_of_inputs = []

    # Test case 1: int padding, 4D input
    input = torch.randn(1, 1, 3, 3).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: tuple padding, 4D input, different paddings
    input = torch.randn(1, 3, 5, 5).numpy()
    padding = (1, 2, 0, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: tuple padding, 3D input
    input = torch.randn(3, 4, 4).numpy()
    padding = (1, 1, 1, 0)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: int padding, 3D input
    input = torch.randn(3, 4, 4).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Float Input
    input = torch.randn(1, 1, 3, 3, dtype=torch.float64).numpy()
    padding = 1
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Smaller padding values
    input = torch.randn(1, 1, 4, 4).numpy()
    padding = (1, 1, 1, 1)
    input_dict = {"padding": padding, "input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReflectionPad2d_2"] = ReflectionPad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReflectionPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad2d_2'.")

check_valid('torch.nn.ReflectionPad2d', generated_inputs['torch.nn.ReflectionPad2d_2'], lib="torch")

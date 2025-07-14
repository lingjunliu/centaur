
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def constantpad2d_inputs():
    list_of_inputs = []

    # Input 1: Basic padding with a single value
    input_tensor = torch.randn(1, 1, 3, 4).numpy()
    padding = (1, 1, 1, 1)
    value = 0.0
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different padding values for each side
    input_tensor = torch.randn(1, 3, 5, 5).numpy()
    padding = (2, 0, 1, 3)
    value = 1.5
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero padding
    input_tensor = torch.randn(2, 2, 4, 6).numpy()
    padding = (0, 0, 0, 0)
    value = 0.0
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large padding values
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    padding = (5, 5, 5, 5)
    value = -1.0
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Padding with a different value
    input_tensor = torch.randn(1, 3, 3, 3).numpy()
    padding = (1, 2, 0, 1)
    value = 2.7
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single channel input
    input_tensor = torch.randn(1, 1, 4, 4).numpy()
    padding = (2, 2, 2, 2)
    value = 0.5
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different batch size
    input_tensor = torch.randn(4, 3, 2, 2).numpy()
    padding = (1, 0, 1, 0)
    value = -0.3
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Non-square input
    input_tensor = torch.randn(1, 1, 3, 5).numpy()
    padding = (1, 1, 2, 2)
    value = 1.0
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger input tensor
    input_tensor = torch.randn(2, 3, 10, 10).numpy()
    padding = (3, 2, 1, 0)
    value = 0.8
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Channel first format
    input_tensor = torch.randn(3, 5, 5).numpy()
    padding = (1, 1, 1, 1)
    value = -0.5
    input_dict = {"padding": padding, "value": float(value), "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConstantPad2d_2"] = constantpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConstantPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad2d_2'.")

check_valid('torch.nn.ConstantPad2d', generated_inputs['torch.nn.ConstantPad2d_2'], lib="torch", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def constantpad2d_inputs():
    list_of_inputs = []

    # Input 1: Integer padding, positive value
    input_tensor = torch.randn(1, 1, 3, 3).numpy()
    padding = 1
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple padding, positive value
    input_tensor = torch.randn(1, 1, 3, 3).numpy()
    padding = (1, 2, 3, 4)
    value = 1.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer padding, negative value
    input_tensor = torch.randn(1, 1, 3, 3).numpy()
    padding = 2
    value = -1.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple padding, negative value
    input_tensor = torch.randn(1, 1, 3, 3).numpy()
    padding = (2, 1, 4, 3)
    value = -2.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different input shape
    input_tensor = torch.randn(2, 3, 5, 5).numpy()
    padding = 1
    value = 0.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different input shape and padding
    input_tensor = torch.randn(3, 1, 4, 6).numpy()
    padding = (0, 1, 2, 0)
    value = 1.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large padding
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    padding = 5
    value = 2.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Padding with zeros
    input_tensor = torch.randn(1, 1, 4, 4).numpy()
    padding = (0, 0, 0, 0)
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single channel image
    input_tensor = torch.randn(1, 1, 3, 3).numpy()
    padding = (1, 1, 1, 1)
    value = 1.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: C,H,W format
    input_tensor = torch.randn(3, 5, 5).numpy()
    padding = 2
    value = 0.7
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Float64 input
    input_tensor = torch.randn(1, 1, 3, 3, dtype=torch.float64).numpy()
    padding = 1
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConstantPad2d_1"] = constantpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConstantPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad2d_1'.")

check_valid('torch.nn.ConstantPad2d', generated_inputs['torch.nn.ConstantPad2d_1'], lib="torch", suffix=1)

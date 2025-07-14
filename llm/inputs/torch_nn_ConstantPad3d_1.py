
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def constantpad3d_inputs():
    list_of_inputs = []

    # Input 1: int padding, simple case
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    padding = 2
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: tuple padding, different values
    input_tensor = torch.randn(1, 1, 3, 3, 3).numpy()
    padding = (1, 2, 0, 1, 2, 0)
    value = 1.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: tuple padding, all zeros
    input_tensor = torch.randn(4, 2, 2, 2, 2).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    value = -1.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int padding, negative value
    input_tensor = torch.randn(1, 3, 5, 7, 9).numpy()
    padding = 1
    value = -2.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: tuple padding, asymmetric padding
    input_tensor = torch.randn(2, 1, 6, 4, 8).numpy()
    padding = (2, 1, 3, 0, 1, 2)
    value = 0.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: small input tensor, int padding
    input_tensor = torch.randn(1, 1, 1, 1, 1).numpy()
    padding = 1
    value = 10.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large padding values, tuple padding
    input_tensor = torch.randn(1, 1, 2, 2, 2).numpy()
    padding = (5, 5, 5, 5, 5, 5)
    value = 0.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different shaped input
    input_tensor = torch.randn(3, 5, 2, 7, 3).numpy()
    padding = (1, 0, 2, 1, 0, 2)
    value = -0.5
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: input with only channel, Din, Hin, Win dimensions
    input_tensor = torch.randn(3, 4, 5, 6).numpy()
    padding = 1
    value = 2.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: input with only channel, Din, Hin, Win dimensions and tuple padding
    input_tensor = torch.randn(3, 4, 5, 6).numpy()
    padding = (1, 1, 2, 2, 0, 0)
    value = 2.0
    input_dict = {"padding": padding, "value": value, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConstantPad3d_1"] = constantpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConstantPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad3d_1'.")

check_valid('torch.nn.ConstantPad3d', generated_inputs['torch.nn.ConstantPad3d_1'], lib="torch", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def constantpad3d_inputs():
    list_of_inputs = []

    # Input 1: Basic padding with a single integer
    input1 = torch.randn(2, 3, 4, 5, 6).numpy()
    padding1 = (2, 2, 2, 2, 2, 2)
    value1 = 0.0
    input_dict1 = {"padding": padding1, "value": value1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different padding on each side
    input2 = torch.randn(1, 1, 3, 3, 3).numpy()
    padding2 = (1, 2, 0, 1, 2, 0)
    value2 = 1.5
    input_dict2 = {"padding": padding2, "value": value2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Zero padding
    input3 = torch.randn(4, 2, 5, 5, 5).numpy()
    padding3 = (0, 0, 0, 0, 0, 0)
    value3 = -1.0
    input_dict3 = {"padding": padding3, "value": value3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Only Channel Din Hin Win
    input4 = torch.randn(3, 4, 5, 6).numpy()
    padding4 = (1, 1, 2, 2, 0, 0)
    value4 = 2.0
    input_dict4 = {"padding": padding4, "value": value4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Negative padding values (should be invalid, but testing it)
    input5 = torch.randn(1, 1, 2, 2, 2).numpy()
    padding5 = (-1, -1, -1, -1, -1, -1)
    value5 = 0.5
    input_dict5 = {"padding": padding5, "value": value5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Large padding values
    input6 = torch.randn(1, 1, 1, 1, 1).numpy()
    padding6 = (10, 10, 10, 10, 10, 10)
    value6 = -0.5
    input_dict6 = {"padding": padding6, "value": value6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: padding with large value
    input7 = torch.randn(2, 3, 4, 5, 6).numpy()
    padding7 = (2, 2, 2, 2, 2, 2)
    value7 = 1000.0
    input_dict7 = {"padding": padding7, "value": value7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: padding with negative value
    input8 = torch.randn(2, 3, 4, 5, 6).numpy()
    padding8 = (2, 2, 2, 2, 2, 2)
    value8 = -1000.0
    input_dict8 = {"padding": padding8, "value": value8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: padding with zero value
    input9 = torch.randn(2, 3, 4, 5, 6).numpy()
    padding9 = (2, 2, 2, 2, 2, 2)
    value9 = 0.0
    input_dict9 = {"padding": padding9, "value": value9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: different padding values, some zero
    input10 = torch.randn(1, 1, 3, 3, 3).numpy()
    padding10 = (0, 2, 0, 1, 1, 0)
    value10 = 1.5
    input_dict10 = {"padding": padding10, "value": value10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: test different input shape
    input11 = torch.randn(1, 3, 1, 1, 1).numpy()
    padding11 = (1, 2, 0, 1, 2, 0)
    value11 = 1.5
    input_dict11 = {"padding": padding11, "value": value11, "input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad3d_2"] = constantpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConstantPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad3d_2'.")

check_valid('torch.nn.ConstantPad3d', generated_inputs['torch.nn.ConstantPad3d_2'], lib="torch", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def replicationpad2d_inputs():
    list_of_inputs = []

    # Input 1: int padding
    input1 = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    padding1 = 2
    input_dict1 = {"padding": (padding1,), "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: tuple padding with different values
    input2 = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    padding2 = (2, 2, 2, 2)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: C, H, W format
    input3 = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3).numpy()
    padding3 = (2, 2, 2, 2)
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: N, C, H, W format, larger values
    input4 = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).numpy()
    padding4 = (2, 2, 2, 2)
    input_dict4 = {"padding": padding4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: N, C, H, W format, padding 0
    input5 = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2).numpy()
    padding5 = (0, 0, 0, 0)
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: C, H, W format, asymmetric padding
    input6 = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).numpy()
    padding6 = (2, 2, 2, 2)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: N, C, H, W format, large padding
    input7 = torch.arange(1, dtype=torch.float32).reshape(1, 1, 1, 1).numpy()
    padding7 = (6, 6, 6, 6)
    input_dict7 = {"padding": padding7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: C, H, W format, int padding
    input8 = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2).numpy()
    padding8 = 2
    input_dict8 = {"padding": (padding8,), "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: N, C, H, W format
    input9 = torch.arange(6, dtype=torch.float32).reshape(1, 2, 1, 3).numpy()
    padding9 = (2, 2, 2, 2)
    input_dict9 = {"padding": padding9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReplicationPad2d_2"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_2'.")

check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_2'], lib="torch", suffix=2)

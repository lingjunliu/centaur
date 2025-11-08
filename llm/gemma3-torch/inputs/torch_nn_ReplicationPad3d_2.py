
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def replicationpad3d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(16, 3, 8, 320, 480)
    padding1 = (3, 3, 6, 6, 1, 1)
    
    input_dict1 = {
        "input": input1.numpy(),
        "padding": padding1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 5, 10, 20, 30)
    padding2 = (2, 2, 2, 2, 2, 2)
    
    input_dict2 = {
        "input": input2.numpy(),
        "padding": padding2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(32, 1, 16, 64, 128)
    padding3 = (1, 2, 0, 1, 3, 4)
    
    input_dict3 = {
        "input": input3.numpy(),
        "padding": padding3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(8, 6, 4, 8, 16)
    padding4 = (0, 0, 0, 0, 0, 0)

    input_dict4 = {
        "input": input4.numpy(),
        "padding": padding4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 4, 5, 10, 15)
    padding5 = (1, 1, 1, 1, 1, 1)
    
    input_dict5 = {
        "input": input5.numpy(),
        "padding": padding5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replicationpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_2'.")


check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_2'], lib="torch", suffix=2)

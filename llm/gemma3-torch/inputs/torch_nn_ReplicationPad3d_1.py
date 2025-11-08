
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def replicationpad3d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(16, 3, 8, 320, 480).numpy()
    padding1 = 3
    input_dict1 = {"padding": padding1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(1, 5, 10, 20, 30).numpy()
    padding2 = (1, 2, 3, 4, 5, 6)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(4, 2, 5, 10, 15).numpy()
    padding3 = 0
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 1, 3, 7, 11).numpy()
    padding4 = (2, 2, 1, 1, 0, 0)
    input_dict4 = {"padding": padding4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(8, 6, 4, 8, 12).numpy()
    padding5 = -1 
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 1, 1, 1).numpy()
    padding6 = (5, 5, 5, 5, 5, 5)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(32, 1, 16, 64, 128).numpy()
    padding7 = 10
    input_dict7 = {"padding": padding7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 4, 6, 9, 12).numpy()
    padding8 = (1, 1, 2, 2, 3, 3)
    input_dict8 = {"padding": padding8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(5, 2, 3, 4, 5).numpy()
    padding9 = 2
    input_dict9 = {"padding": padding9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(1, 3, 7, 11, 13).numpy()
    padding10 = (0, 0, 0, 0, 0, 0)
    input_dict10 = {"padding": padding10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replicationpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_1'.")


check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_1'], lib="torch", suffix=1)

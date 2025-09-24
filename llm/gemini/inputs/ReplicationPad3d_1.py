
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def ReplicationPad3d_inputs():
    list_of_inputs = []

    # Input 1: Integer padding, 5D input
    input_1 = torch.randn(2, 3, 4, 5, 6).numpy()
    padding_1 = 2
    input_dict_1 = {"input": input_1, "padding": padding_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Tuple padding, 5D input
    input_2 = torch.randn(1, 1, 3, 3, 3).numpy()
    padding_2 = (1, 2, 0, 1, 2, 0)
    input_dict_2 = {"input": input_2, "padding": padding_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Integer padding, 4D input
    input_3 = torch.randn(3, 4, 5, 6).numpy()
    padding_3 = 1
    input_dict_3 = {"input": input_3, "padding": padding_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tuple padding, 4D input
    input_4 = torch.randn(2, 3, 2, 2).numpy()
    padding_4 = (0, 1, 1, 0, 2, 1)
    input_dict_4 = {"input": input_4, "padding": padding_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different input size
    input_5 = torch.randn(4, 5, 10, 12, 14).numpy()
    padding_5 = 3
    input_dict_5 = {"input": input_5, "padding": padding_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Zero Padding
    input_6 = torch.randn(1, 2, 3, 4, 5).numpy()
    padding_6 = 0
    input_dict_6 = {"input": input_6, "padding": padding_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Asymmetric Padding
    input_7 = torch.randn(1, 2, 3, 4, 5).numpy()
    padding_7 = (1, 0, 2, 1, 0, 2)
    input_dict_7 = {"input": input_7, "padding": padding_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReplicationPad3d_1"] = ReplicationPad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_1'.")

check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_1'], lib="torch")

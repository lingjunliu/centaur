
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def replicationpad2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int padding
    input1 = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    padding1 = 2
    input_dict1 = {"input": input1, "padding": padding1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different padding on each side
    input2 = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    padding2 = (1, 1, 2, 0)
    input_dict2 = {"input": input2, "padding": padding2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3:  C,H,W format
    input3 = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    padding3 = 1
    input_dict3 = {"input": input3, "padding": padding3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: N,C,H,W with different padding
    input4 = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4)
    padding4 = (0, 2, 1, 1)
    input_dict4 = {"input": input4, "padding": padding4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5:  C,H,W format with tuple padding
    input5 = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    padding5 = (1, 0, 2, 1)
    input_dict5 = {"input": input5, "padding": padding5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Larger padding
    input6 = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    padding6 = 5
    input_dict6 = {"input": input6, "padding": padding6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Non-square input
    input7 = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    padding7 = (2, 1, 0, 1)
    input_dict7 = {"input": input7, "padding": padding7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Small input and different padding
    input8 = np.array([[[[1.0]]]], dtype=np.float32)
    padding8 = (2, 3, 1, 0)
    input_dict8 = {"input": input8, "padding": padding8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Float64 input
    input9 = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
    padding9 = 1
    input_dict9 = {"input": input9, "padding": padding9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Multi-channel input
    input10 = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)
    padding10 = (1, 0, 0, 1)
    input_dict10 = {"input": input10, "padding": padding10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Padding with zero
    input11 = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    padding11 = 0
    input_dict11 = {"input": input11, "padding": padding11}
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    # Input 12: Large single padding
    input12 = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    padding12 = 10
    input_dict12 = {"input": input12, "padding": padding12}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReplicationPad2d_1"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReplicationPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_1'.")

check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_1'], lib="torch", suffix=1)

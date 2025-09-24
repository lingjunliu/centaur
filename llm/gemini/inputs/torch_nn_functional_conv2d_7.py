
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(1, 3, 32, 32).numpy()
    weight1 = torch.randn(5, 3, 5, 5).numpy()
    bias1 = torch.randn(5).numpy()
    input_dict1 = {'input': input1, 'weight': weight1, 'bias': bias1, 'stride': 1, 'padding': 0, 'dilation': (1, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = torch.randn(2, 4, 28, 28).numpy()
    weight2 = torch.randn(8, 4, 3, 3).numpy()
    bias2 = torch.randn(8).numpy()
    input_dict2 = {'input': input2, 'weight': weight2, 'bias': bias2, 'stride': 2, 'padding': 1, 'dilation': (1, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = torch.randn(1, 16, 64, 64).numpy()
    weight3 = torch.randn(32, 16, 7, 7).numpy()
    bias3 = torch.randn(32).numpy()
    input_dict3 = {'input': input3, 'weight': weight3, 'bias': bias3, 'stride': 1, 'padding': 3, 'dilation': (2, 2), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = torch.randn(4, 3, 16, 16).numpy()
    weight4 = torch.randn(6, 3, 3, 3).numpy()
    bias4 = torch.randn(6).numpy()
    input_dict4 = {'input': input4, 'weight': weight4, 'bias': bias4, 'stride': 2, 'padding': 1, 'dilation': (1, 2), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = torch.randn(1, 8, 128, 128).numpy()
    weight5 = torch.randn(16, 8, 5, 5).numpy()
    bias5 = torch.randn(16).numpy()
    input_dict5 = {'input': input5, 'weight': weight5, 'bias': bias5, 'stride': 4, 'padding': 2, 'dilation': (2, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = torch.randn(2, 24, 24, 24).numpy()
    weight6 = torch.randn(48, 24, 3, 3).numpy()
    bias6 = torch.randn(48).numpy()
    input_dict6 = {'input': input6, 'weight': weight6, 'bias': bias6, 'stride': 1, 'padding': 1, 'dilation': (1, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = torch.randn(1, 32, 48, 48).numpy()
    weight7 = torch.randn(64, 32, 5, 5).numpy()
    bias7 = torch.randn(64).numpy()
    input_dict7 = {'input': input7, 'weight': weight7, 'bias': bias7, 'stride': 2, 'padding': 2, 'dilation': (2, 2), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = torch.randn(4, 12, 20, 20).numpy()
    weight8 = torch.randn(24, 12, 3, 3).numpy()
    bias8 = torch.randn(24).numpy()
    input_dict8 = {'input': input8, 'weight': weight8, 'bias': bias8, 'stride': 1, 'padding': 0, 'dilation': (3, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = torch.randn(1, 64, 16, 16).numpy()
    weight9 = torch.randn(128, 64, 1, 1).numpy()
    bias9 = torch.randn(128).numpy()
    input_dict9 = {'input': input9, 'weight': weight9, 'bias': bias9, 'stride': 1, 'padding': 0, 'dilation': (1, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10
    input10 = torch.randn(1, 3, 16, 16).numpy()
    weight10 = torch.randn(16, 3, 3, 3).numpy()
    bias10 = torch.randn(16).numpy()
    input_dict10 = {'input': input10, 'weight': weight10, 'bias': bias10, 'stride': 2, 'padding': 1, 'dilation': (1, 1), 'groups': 1}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv2d_7"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv2d_7'.")

check_valid('torch.nn.functional.conv2d', generated_inputs['torch.nn.functional.conv2d_7'], lib="torch", suffix=7)

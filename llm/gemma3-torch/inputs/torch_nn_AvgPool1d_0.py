
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool1d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(1, 3, 10).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(1, 5, 15).numpy()
    input_dict2 = {
        "kernel_size": 5,
        "stride": 3,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(2, 4, 8).numpy()
    input_dict3 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(1, 2, 20).numpy()
    input_dict4 = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(3, 1, 12).numpy()
    input_dict5 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 4, 7).numpy()
    input_dict6 = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 3, 11).numpy()
    input_dict7 = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(1, 6, 18).numpy()
    input_dict8 = {
        "kernel_size": 5,
        "stride": 5,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(4, 2, 9).numpy()
    input_dict9 = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d"] = avgpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d'.")


check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool1d_inputs():
    list_of_inputs = []
    
    input_arr = np.arange(1, 8, dtype=np.float32).reshape(1, 1, 7)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.linspace(-5, 5, num=48, dtype=np.float32).reshape(2, 3, 8)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.arange(30, dtype=np.float64).reshape(3, 1, 10)
    input_dict = {
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[0.5, -1.0, 2.0, -2.5, 3.0],
                           [1.5, 0.0, -0.5, 0.5, -1.5],
                           [-2.0, 2.0, -3.0, 3.0, 0.0],
                           [4.0, -4.0, 1.0, -1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.linspace(-1, 1, num=80, dtype=np.float32).reshape(2, 2, 20)
    input_dict = {
        "kernel_size": 7,
        "stride": 7,
        "padding": 3,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.arange(240, dtype=np.float32).reshape(5, 4, 12) / 10.0 - 12.0
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[1.0, -1.0, 2.0, -2.0],
                           [0.5, -0.5, 1.5, -1.5],
                           [3.0, -3.0, 0.0, 0.0]]], dtype=np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.linspace(0, 8, num=18, dtype=np.float32).reshape(2, 1, 9)
    input_dict = {
        "kernel_size": 8,
        "stride": 2,
        "padding": 4,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.linspace(-10.0, 10.0, num=50, dtype=np.float64).reshape(2, 25)
    input_dict = {
        "kernel_size": 9,
        "stride": 4,
        "padding": 4,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.linspace(-3, 3, num=10, dtype=np.float32).reshape(1, 2, 5)
    input_dict = {
        "kernel_size": 3,
        "stride": 5,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d_1"] = avgpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_1'.")


check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_1'], lib="torch", suffix=1)

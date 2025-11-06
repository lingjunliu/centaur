
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool1d_inputs():
    list_of_inputs = []

    input_arr = np.array([[[1., 2., 3., 4., 5., 6., 7.]]], dtype=np.float32)
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(16, dtype=np.float32).reshape(2, 8)
    input_dict = {
        "kernel_size": (2,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 3, 7).astype(np.float32)
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.uniform(-3, 3, size=(1, 2, 10)).astype(np.float32)
    input_dict = {
        "kernel_size": (4,),
        "stride": (3,),
        "padding": (2,),
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.linspace(-5, 5, 10, dtype=np.float32).reshape(1, 10)
    input_dict = {
        "kernel_size": (5,),
        "stride": (1,),
        "padding": (2,),
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(4, 2, 15).astype(np.float64)
    input_dict = {
        "kernel_size": (3,),
        "stride": (3,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[-1.]], [[2.]], [[-3.]]], dtype=np.float32).reshape(3, 1)
    input_dict = {
        "kernel_size": (1,),
        "stride": (1,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 1, 9).astype(np.float32)
    input_dict = {
        "kernel_size": (2,),
        "stride": (3,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (np.arange(48, dtype=np.float16).reshape(4, 12) - 24) / 10.0
    input_dict = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (1,),
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(1, 3, 20).astype(np.float32)
    input_dict = {
        "kernel_size": (7,),
        "stride": (2,),
        "padding": (3,),
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.uniform(-1, 1, size=(2, 3, 5)).astype(np.float32)
    input_dict = {
        "kernel_size": (5,),
        "stride": (5,),
        "padding": (0,),
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[0., -1., 2., -3., 4., -5.]]], dtype=np.float32)
    input_dict = {
        "kernel_size": (3,),
        "stride": (4,),
        "padding": (1,),
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d_2"] = avgpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_2'.")


check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_2'], lib="torch", suffix=2)

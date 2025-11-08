
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool3d_inputs_2():
    list_of_inputs = []

    inp = np.random.randn(1, 1, 4, 4, 4).astype(np.float32)
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(2, 3, 5, 6, 7).astype(np.float64)
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (1, 1, 2),
        "padding": (1, 0, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 12,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(3, 10, 12, 8).astype(np.float32)
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": (2, 1, 2),
        "padding": (0, 1, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 12,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.linspace(-2.0, 2.0, num=54).reshape(1, 2, 3, 3, 3).astype(np.float32)
    input_dict = {
        "kernel_size": (1, 2, 2),
        "stride": (1, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": data
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(2, 4, 7, 7, 7).astype(np.float32)
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (3, 3, 3),
        "padding": (0, 0, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 8,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(1, 1, 5, 5, 5).astype(np.float64)
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 27,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(1, 2, 8, 3, 5).astype(np.float32)
    input_dict = {
        "kernel_size": (4, 1, 2),
        "stride": (2, 1, 2),
        "padding": (1, 0, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(2, 3, 3, 3).astype(np.float32)
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 8,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.random.randn(3, 1, 9, 5, 7).astype(np.float32)
    input_dict = {
        "kernel_size": (5, 2, 3),
        "stride": (3, 2, 2),
        "padding": (2, 0, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 30,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (np.random.randn(1, 1, 6, 6, 6) * 0.1).astype(np.float32)
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (3, 3, 3),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 27,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.zeros((4, 2, 4, 5, 6), dtype=np.float32)
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": (1, 2, 2),
        "padding": (0, 1, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 12,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (np.random.rand(1, 3, 10, 8, 6).astype(np.float64) - 0.5) * 2.0
    input_dict = {
        "kernel_size": (3, 4, 2),
        "stride": (2, 3, 2),
        "padding": (1, 1, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 24,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool3d_2"] = avgpool3d_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool3d_2'.")


check_valid('torch.nn.AvgPool3d', generated_inputs['torch.nn.AvgPool3d_2'], lib="torch", suffix=2)

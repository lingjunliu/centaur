
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def avgpool2d_inputs_v2():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(1, 1, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": np.int64(4),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(2, 3, 5, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int32(6),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn(3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": np.int64(9),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(4, 2, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": np.int32(3),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(2, 1, 9, 5, dtype=torch.float16).numpy()
    input_dict = {
        "kernel_size": (5, 1),
        "stride": (3, 1),
        "padding": (2, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": np.int64(5),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.randn(1, 3, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 3),
        "stride": (2, 3),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int32(6),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.ones(1, 1, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int64(9),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(1, 1, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": np.int32(4),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(2, 4, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 5),
        "stride": (3, 4),
        "padding": (1, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": np.int64(20),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.randn(1, 10, 7, dtype=torch.double).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int32(4),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(1, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (3, 3),
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": np.int64(4),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = torch.randn(3, 3, 11, 13, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (5, 3),
        "stride": (2, 2),
        "padding": (2, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int32(15),
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_2"] = avgpool2d_inputs_v2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_2'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_2'], lib="torch", suffix=2)

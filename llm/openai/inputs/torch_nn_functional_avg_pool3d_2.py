
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int64(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn((2, 3, 5, 6, 7), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": np.int32(5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.linspace(-5.0, 5.0, steps=54, dtype=torch.float64).reshape(1, 2, 3, 3, 3).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (1, 2, 2),
        "stride": (1, 1, 1),
        "padding": (0, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": np.int64(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn((4, 1, 8, 1, 5), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (2, 1, 5),
        "stride": (2, 1, 5),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = (torch.randn((1, 4, 10, 9, 8), dtype=torch.float32) * 2.0 + 1.0).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (4, 3, 2),
        "stride": (3, 2, 2),
        "padding": (1, 1, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": np.int64(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.linspace(-1.0, 1.0, steps=3*2*7*7*7, dtype=torch.float32).reshape(3, 2, 7, 7, 7).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (3, 1, 1),
        "stride": (1, 2, 3),
        "padding": (1, 0, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": np.int32(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.arange(25, dtype=torch.float32).reshape(1, 1, 1, 5, 5).numpy() - 12.0
    input_dict = {
        "input": input_arr,
        "kernel_size": (1, 3, 3),
        "stride": (1, 2, 2),
        "padding": (0, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": np.int64(9),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn((2, 5, 6, 4, 3), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (2, 2, 3),
        "stride": (2, 2, 1),
        "padding": (0, 0, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": np.int32(6),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.randn((1, 3, 9, 9, 9), dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (5, 5, 5),
        "stride": (4, 4, 4),
        "padding": (2, 2, 2),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": np.int64(7),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = (torch.rand((5, 3, 3, 3, 3), dtype=torch.float32) * 2.0 - 1.0).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": np.int32(10),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.tensor([[[[[1., 2.],
                                 [3., 4.]],
                                [[5., 6.],
                                 [7., 8.]]],
                               [[[-1., -2.],
                                 [-3., -4.]],
                                [[-5., -6.],
                                 [-7., -8.]]]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": np.int64(8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn((2, 2, 4, 3, 2), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": (2, 3, 2),
        "stride": (1, 2, 1),
        "padding": (0, 1, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": np.int32(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool3d_2"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_2'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_2'], lib="torch", suffix=2)

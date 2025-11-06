
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lppool2d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(1, 1, 6, 6).numpy()
    input_dict = {
        "norm_type": 2.0,
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.rand(2, 3, 7, 7).numpy()
    input_dict = {
        "norm_type": 1.0,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.randn(2, 4, 9, 5).numpy()
    input_dict = {
        "norm_type": 3.0,
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.rand(3, 2, 11, 10).numpy()
    input_dict = {
        "norm_type": 1.2,
        "kernel_size": (5, 4),
        "stride": (2, 2),
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(3, 2, 4, 4).numpy()
    input_dict = {
        "norm_type": 4.0,
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = torch.rand(3, 8, 9).numpy()
    input_dict = {
        "norm_type": 2.5,
        "kernel_size": (2, 3),
        "stride": (1, 2),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.randn(1, 8, 9, 9).numpy()
    input_dict = {
        "norm_type": 6.0,
        "kernel_size": (3, 3),
        "stride": (3, 3),
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.rand(5, 10, 3).numpy()
    input_dict = {
        "norm_type": 7.5,
        "kernel_size": (2, 1),
        "stride": (1, 1),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(4, 3, 12, 12, dtype=torch.float64).numpy()
    input_dict = {
        "norm_type": 5.0,
        "kernel_size": (4, 4),
        "stride": (2, 2),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = torch.randn(2, 2, 5, 9).numpy()
    input_dict = {
        "norm_type": 8.0,
        "kernel_size": (2, 3),
        "stride": (2, 3),
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(3, 6, 10, 20).numpy()
    input_dict = {
        "norm_type": 10.0,
        "kernel_size": (3, 5),
        "stride": (1, 3),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input_arr = torch.rand(2, 5, 6).numpy()
    input_dict = {
        "norm_type": 0.5,
        "kernel_size": (2, 2),
        "stride": (2, 1),
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LPPool2d_2"] = lppool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LPPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LPPool2d_2'.")


check_valid('torch.nn.LPPool2d', generated_inputs['torch.nn.LPPool2d_2'], lib="torch", suffix=2)

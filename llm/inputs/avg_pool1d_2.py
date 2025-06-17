
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 10).numpy()
    kernel_size1 = (2,)
    stride1 = (2,)
    padding1 = (0,)
    ceil_mode1 = False
    count_include_pad1 = True

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "ceil_mode": ceil_mode1,
        "count_include_pad": count_include_pad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 15).numpy()
    kernel_size2 = (3,)
    stride2 = (1,)
    padding2 = (1,)
    ceil_mode2 = True
    count_include_pad2 = False

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "count_include_pad": count_include_pad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool1d_2"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_2'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_2'], lib="torch")

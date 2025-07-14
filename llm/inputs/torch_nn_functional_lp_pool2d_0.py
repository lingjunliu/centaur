
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import torch.nn.functional as F

def lp_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(1, 1, 4, 4).numpy()
    kernel_size1 = (2, 2)
    stride1 = (2, 2)
    padding1 = (0, 0)
    ceil_mode1 = False
    p1 = 2

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "ceil_mode": ceil_mode1,
        "p": p1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = torch.randn(1, 3, 8, 8).numpy()
    kernel_size2 = (3, 3)
    stride2 = (1, 1)
    padding2 = (1, 1)
    ceil_mode2 = True
    p2 = 3

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "p": p2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = torch.randn(2, 3, 16, 16).numpy()
    kernel_size3 = (4, 4)
    stride3 = (4, 4)
    padding3 = (0, 0)
    ceil_mode3 = False
    p3 = 3

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "ceil_mode": ceil_mode3,
        "p": p3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = torch.randn(1, 1, 5, 5).numpy()
    kernel_size4 = (2, 2)
    stride4 = (1, 1)
    padding4 = (1, 1)
    ceil_mode4 = True
    p4 = 2

    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "ceil_mode": ceil_mode4,
        "p": p4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = torch.randn(1, 3, 7, 7).numpy()
    kernel_size5 = (3, 3)
    stride5 = (2, 2)
    padding5 = (1, 1)
    ceil_mode5 = False
    p5 = 1

    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "ceil_mode": ceil_mode5,
        "p": p5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = torch.randn(2, 1, 9, 9).numpy()
    kernel_size6 = (4, 4)
    stride6 = (3, 3)
    padding6 = (2, 2)
    ceil_mode6 = True
    p6 = 2

    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "ceil_mode": ceil_mode6,
        "p": p6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = torch.randn(1, 3, 11, 11).numpy()
    kernel_size7 = (5, 5)
    stride7 = (1, 1)
    padding7 = (0, 0)
    ceil_mode7 = False
    p7 = 3

    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "ceil_mode": ceil_mode7,
        "p": p7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = torch.randn(2, 1, 6, 6).numpy()
    kernel_size8 = (1, 1)
    stride8 = (1, 1)
    padding8 = (0, 0)
    ceil_mode8 = True
    p8 = 2

    input_dict8 = {
        "input": input8,
        "kernel_size": kernel_size8,
        "stride": stride8,
        "padding": padding8,
        "ceil_mode": ceil_mode8,
        "p": p8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = torch.randn(1, 3, 10, 10).numpy()
    kernel_size9 = (2, 2)
    stride9 = (2, 2)
    padding9 = (1, 1)
    ceil_mode9 = False
    p9 = 1

    input_dict9 = {
        "input": input9,
        "kernel_size": kernel_size9,
        "stride": stride9,
        "padding": padding9,
        "ceil_mode": ceil_mode9,
        "p": p9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = torch.randn(2, 1, 8, 8).numpy()
    kernel_size10 = (3, 3)
    stride10 = (1, 1)
    padding10 = (1, 1)
    ceil_mode10 = True
    p10 = 2

    input_dict10 = {
        "input": input10,
        "kernel_size": kernel_size10,
        "stride": stride10,
        "padding": padding10,
        "ceil_mode": ceil_mode10,
        "p": p10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.lp_pool2d"] = lp_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.lp_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.lp_pool2d'.")

check_valid('torch.nn.functional.lp_pool2d', generated_inputs['torch.nn.functional.lp_pool2d'], lib="torch", suffix=0)

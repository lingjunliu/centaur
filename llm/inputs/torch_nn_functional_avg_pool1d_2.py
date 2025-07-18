
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool1d_inputs():
    """
    Generates a list of valid inputs for torch.nn.functional.avg_pool1d.
    The 'divisor_override' parameter is omitted to resolve the TypeError,
    as the execution environment's PyTorch version does not support it.
    """
    list_of_inputs = []

    # Case 1: Basic 3D input (N, C, L)
    input_dict1 = {
        'input': torch.arange(0, 16, dtype=torch.float32).reshape(1, 2, 8).numpy(),
        'kernel_size': (2,),
        'stride': (2,),
        'padding': (0,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Basic 2D input (N, L), treated as (N, 1, L)
    input_dict2 = {
        'input': torch.randn(10, 20).numpy(),
        'kernel_size': (4,),
        'stride': (4,),
        'padding': (0,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Overlapping pooling (stride < kernel_size)
    input_dict3 = {
        'input': torch.ones(1, 1, 10, dtype=torch.float32).numpy(),
        'kernel_size': (4,),
        'stride': (1,),
        'padding': (0,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: With non-zero padding
    input_dict4 = {
        'input': torch.arange(0, 8, dtype=torch.float32).reshape(1, 1, 8).numpy(),
        'kernel_size': (3,),
        'stride': (1,),
        'padding': (1,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: With padding but count_include_pad=False
    input_dict5 = {
        'input': torch.arange(0, 8, dtype=torch.float32).reshape(1, 1, 8).numpy(),
        'kernel_size': (3,),
        'stride': (1,),
        'padding': (1,),
        'ceil_mode': False,
        'count_include_pad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: With ceil_mode=True for non-perfectly-divisible input
    input_dict6 = {
        'input': torch.randn(1, 1, 7).numpy(),
        'kernel_size': (3,),
        'stride': (3,),
        'padding': (0,),
        'ceil_mode': True,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: Strided pooling (stride > kernel_size)
    input_dict7 = {
        'input': torch.arange(0, 12, dtype=torch.float32).reshape(1, 1, 12).numpy(),
        'kernel_size': (2,),
        'stride': (3,),
        'padding': (0,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Case 8: Large input with float64 dtype and some padding
    input_dict8 = {
        'input': np.random.rand(4, 16, 256).astype(np.float64),
        'kernel_size': (8,),
        'stride': (8,),
        'padding': (2,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Case 9: "SAME" padding simulation
    input_dict9 = {
        'input': torch.randn(2, 3, 16).numpy(),
        'kernel_size': (5,),
        'stride': (1,),
        'padding': (2,),
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Case 10: All options enabled with float64
    input_dict10 = {
        'input': torch.randn(1, 2, 9).numpy().astype(np.float64),
        'kernel_size': (4,),
        'stride': (2,),
        'padding': (1,),
        'ceil_mode': True,
        'count_include_pad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool1d_2"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_2'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_2'], lib="torch", suffix=2)

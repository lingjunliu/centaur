
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def lp_pool2d_inputs():
    list_of_inputs = []

    # The error `TypeError: lp_pool2d() takes from 3 to 5 positional arguments but 6 were given`
    # indicates the function was called with too many arguments. The user's requested signature
    # `{'input': 'tensor', 'kernel_size': 'tuple', 'stride': 'tuple', 'padding': 'tuple', 'ceil_mode': 'boolean', 'p': 'integer'}`
    # contains 6 arguments.
    # The actual PyTorch API is `lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)`.
    # It does not accept `padding` and uses `norm_type` instead of `p`.
    # The following inputs are generated for the *actual* API to resolve the TypeError.
    # This requires omitting the `padding` key and using `norm_type`.

    # Input 1: Basic case with 5 arguments
    input_dict1 = {
        'input': torch.randn(1, 3, 10, 10).numpy(),
        'norm_type': 2.0,
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D input tensor
    input_dict2 = {
        'input': torch.randn(3, 8, 8).numpy(),
        'norm_type': 1.0,
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Minimal arguments (3)
    input_dict3 = {
        'input': torch.arange(1., 50.).reshape(1, 1, 7, 7).numpy(),
        'norm_type': 3.0,
        'kernel_size': (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: ceil_mode=True
    input_dict4 = {
        'input': torch.randn(1, 2, 9, 9).numpy(),
        'norm_type': 2.0,
        'kernel_size': (3, 3),
        'stride': (2, 2),
        'ceil_mode': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4 arguments (omitting ceil_mode)
    input_dict5 = {
        'input': torch.randn(1, 1, 16, 16).numpy(),
        'norm_type': 1.0,
        'kernel_size': (4, 4),
        'stride': (4, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Non-square kernel and stride
    input_dict6 = {
        'input': torch.randn(2, 4, 20, 15).numpy(),
        'norm_type': 4.0,
        'kernel_size': (3, 2),
        'stride': (2, 1),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger norm_type
    input_dict7 = {
        'input': torch.randn(2, 4, 16, 16).numpy(),
        'norm_type': 5.0,
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Stride > kernel_size
    input_dict8 = {
        'input': torch.randn(1, 1, 10, 10).numpy(),
        'norm_type': 2.0,
        'kernel_size': (2, 2),
        'stride': (3, 3),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Input with negative values
    input_dict9 = {
        'input': (torch.randn(1, 3, 8, 8) * 2 - 1).numpy(),
        'norm_type': 2.0,
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: float64 dtype
    input_dict10 = {
        'input': torch.randn(1, 3, 10, 10, dtype=torch.float64).numpy(),
        'norm_type': 2.0,
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'ceil_mode': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.lp_pool2d"] = lp_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.lp_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.lp_pool2d'.")

check_valid('torch.nn.functional.lp_pool2d', generated_inputs['torch.nn.functional.lp_pool2d'], lib="torch", suffix=0)

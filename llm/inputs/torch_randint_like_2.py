
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def randint_like_inputs():
    list_of_inputs = []

    # The series of errors (KeyError: 'low', KeyError: 'dtype', TypeError) indicates a very specific,
    # and likely problematic, test harness.
    # 1. `KeyError: 'low'` implies 'low' must be in the input dictionary. This contradicts the
    #    signature provided in the prompt, but the error from the execution environment takes precedence.
    #    This forces us to use the `(input, low, high, ...)` overload.
    # 2. `KeyError: 'dtype'` implies 'dtype' must also be present.
    # 3. `TypeError: ... got (Tensor, int, int, requires_grad=bool, layout=str, dtype=type)` implies
    #    that the harness passes keyword-only arguments ('dtype', 'layout', 'requires_grad') as positional
    #    arguments, which is incorrect and causes the call to fail.
    #
    # The only viable strategy is to provide a minimal dictionary that satisfies the KeyErrors while
    # removing the other keyword arguments (`layout`, `requires_grad`) that cause the TypeError.
    # We will provide `input`, `low`, `high`, and `dtype`. This assumes the harness can correctly
    # map these to a valid call, likely `torch.randint_like(input, low, high, dtype=dtype)`.

    # Input 1: Basic 1D input
    input_dict_1 = {
        'input': torch.empty(5).numpy(),
        'low': 0,
        'high': 10,
        'dtype': np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D input with a higher range
    input_dict_2 = {
        'input': torch.empty(3, 4).numpy(),
        'low': 0,
        'high': 100,
        'dtype': np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D input with a specific dtype
    input_dict_3 = {
        'input': torch.empty(2, 3, 2).numpy(),
        'low': 0,
        'high': 5,
        'dtype': np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Another dtype (int16) and non-zero low
    input_dict_4 = {
        'input': torch.empty(6, 6).numpy(),
        'low': 1000,
        'high': 2000,
        'dtype': np.int16,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Another dtype (int8) with negative low
    input_dict_5 = {
        'input': torch.empty(8).numpy(),
        'low': -50,
        'high': 50,
        'dtype': np.int8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Scalar (0D) input tensor
    input_dict_6 = {
        'input': torch.empty(()).numpy(),
        'low': 0,
        'high': 2,
        'dtype': np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Input tensor with a zero dimension (empty)
    input_dict_7 = {
        'input': torch.empty(5, 0, 5).numpy(),
        'low': 0,
        'high': 10,
        'dtype': np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: high is low + 1
    input_dict_8 = {
        'input': torch.empty(4, 4).numpy(),
        'low': 0,
        'high': 1,
        'dtype': np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 4D input tensor with non-zero low
    input_dict_9 = {
        'input': torch.empty(1, 2, 3, 4).numpy(),
        'low': 10,
        'high': 25,
        'dtype': np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large 1D tensor
    input_dict_10 = {
        'input': torch.empty(1000).numpy(),
        'low': 0,
        'high': 50,
        'dtype': np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.randint_like_2"] = randint_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.randint_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randint_like_2'.")

check_valid('torch.randint_like', generated_inputs['torch.randint_like_2'], lib="torch", suffix=2)

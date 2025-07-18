
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def repeat_interleave_inputs():
    list_of_inputs = []

    # The user is facing a TypeError because the `out` argument is not supported
    # when the `repeats` argument is a tensor. The provided signature is
    # {'input': 'tensor', 'repeats': 'tensor', 'dim': 'integer', 'out': 'tensor'}.
    # The only way to resolve the TypeError while adhering to the signature's
    # type for `repeats` is to omit the `out` argument. The previous `KeyError`
    # might have been due to inconsistent dictionaries (some with `out`, some without).
    # This solution consistently omits `out` to fix the TypeError.

    # Case 1: 1D input, scalar repeats tensor
    input_dict_1 = {
        'input': torch.tensor([1, 2, 3]).numpy(),
        'repeats': torch.tensor(2).numpy(),
        'dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: 2D input, scalar repeats tensor, dim=0
    input_dict_2 = {
        'input': torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).numpy(),
        'repeats': torch.tensor(3).numpy(),
        'dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: 2D input, scalar repeats tensor, dim=1
    input_dict_3 = {
        'input': torch.tensor([[1, 2], [3, 4]]).numpy(),
        'repeats': torch.tensor(2).numpy(),
        'dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: 1D input with 1D repeats tensor
    input_dict_4 = {
        'input': torch.tensor([10, 20, 30]).numpy(),
        'repeats': torch.tensor([1, 2, 3]).numpy(),
        'dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Case 5: 2D input with 1D repeats tensor, dim=0
    input_dict_5 = {
        'input': torch.tensor([[10, 20], [30, 40]]).numpy(),
        'repeats': torch.tensor([2, 1]).numpy(),
        'dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: 2D input with 1D repeats tensor, dim=1, float64 dtype
    input_dict_6 = {
        'input': torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64).numpy(),
        'repeats': torch.tensor([1, 3, 2]).numpy(),
        'dim': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: 3D input, scalar repeats, negative dim
    input_dict_7 = {
        'input': torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).numpy(),
        'repeats': torch.tensor(2).numpy(),
        'dim': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Repeats tensor containing zero
    input_dict_8 = {
        'input': torch.tensor([10, 20, 30]).numpy(),
        'repeats': torch.tensor([2, 0, 1]).numpy(),
        'dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: 0-dim (scalar) input tensor
    input_dict_9 = {
        'input': torch.tensor(42).numpy(),
        'repeats': torch.tensor(5).numpy(),
        'dim': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: 4D input with float16 dtype
    input_dict_10 = {
        'input': torch.ones(2, 1, 3, 1, dtype=torch.float16).numpy(),
        'repeats': torch.tensor(4).numpy(),
        'dim': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.repeat_interleave_2"] = repeat_interleave_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.repeat_interleave_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.repeat_interleave_2'.")

check_valid('torch.repeat_interleave', generated_inputs['torch.repeat_interleave_2'], lib="torch", suffix=2)

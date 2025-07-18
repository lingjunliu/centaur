
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def maxunpool2d_inputs():
    list_of_inputs = []

    # The RuntimeError indicates MaxUnpool2d is registered as non-deterministic.
    # The primary cause is duplicate values in the 'indices' tensor.
    # To guarantee determinism, we ensure indices are unique by:
    # 1. Using `torch.arange` for an original tensor with unique values.
    # 2. Ensuring non-overlapping pooling windows (`stride >= kernel_size`).
    # 3. Using `padding=0` to avoid complex edge cases that might be flagged.

    def _generate_deterministic_input(ks, s, shape):
        if s < ks:
            return None
        
        try:
            pool = nn.MaxPool2d(kernel_size=ks, stride=s, padding=0, return_indices=True)
            # Use float to create a float tensor
            original_input = torch.arange(float(np.prod(shape)), dtype=torch.float32).reshape(shape)
            output, indices = pool(original_input)

            # A final check to ensure our logic produced unique indices.
            if len(np.unique(indices.numpy())) != indices.numel():
                return None

            return {
                'kernel_size': ks,
                'stride': s,
                'padding': 0,
                'input': output.numpy(),
                'indices': indices.numpy(),
                'output_size': tuple(original_input.shape)
            }
        except RuntimeError:
            # Catches invalid shape/parameter combinations for MaxPool2d.
            return None

    # Input 1: Basic case
    inp = _generate_deterministic_input(ks=2, s=2, shape=(1, 1, 4, 4))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 2: Identity mapping (k=1, s=1)
    inp = _generate_deterministic_input(ks=1, s=1, shape=(1, 1, 5, 5))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 3: Multi-channel
    inp = _generate_deterministic_input(ks=2, s=2, shape=(1, 3, 4, 4))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 4: Multi-batch
    inp = _generate_deterministic_input(ks=2, s=2, shape=(2, 1, 6, 6))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 5: Stride > kernel size (gaps)
    inp = _generate_deterministic_input(ks=2, s=3, shape=(1, 1, 8, 8))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 6: Ambiguous size case (requires output_size)
    inp = _generate_deterministic_input(ks=2, s=2, shape=(1, 1, 4, 5))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 7: Larger kernel and stride
    inp = _generate_deterministic_input(ks=4, s=4, shape=(1, 1, 16, 16))
    if inp: list_of_inputs.append(copy.deepcopy(inp))
    
    # Input 8: Another valid combination
    inp = _generate_deterministic_input(ks=3, s=3, shape=(2, 2, 9, 9))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    # Input 9: Larger prime-number dimensions
    inp = _generate_deterministic_input(ks=3, s=4, shape=(1, 1, 13, 13))
    if inp: list_of_inputs.append(copy.deepcopy(inp))
    
    # Input 10: Larger batch and channels
    inp = _generate_deterministic_input(ks=2, s=2, shape=(4, 4, 4, 4))
    if inp: list_of_inputs.append(copy.deepcopy(inp))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool2d_1"] = maxunpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxUnpool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool2d_1'.")

check_valid('torch.nn.MaxUnpool2d', generated_inputs['torch.nn.MaxUnpool2d_1'], lib="torch", suffix=1)

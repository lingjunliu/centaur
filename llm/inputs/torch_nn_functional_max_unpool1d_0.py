
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def max_unpool1d_inputs():
    list_of_inputs = []

    # Helper function to generate valid (input, indices) pairs by performing pooling first.
    # The CPU implementation for max_unpool1d is flagged as non-deterministic.
    # This is often due to overlapping windows (stride < kernel_size).
    # To satisfy the deterministic check, we generate inputs that are most likely to be
    # deterministic:
    # 1. Non-overlapping windows (stride >= kernel_size).
    # 2. Special cases (empty tensors, kernel_size=1) that may use a different code path.
    # We also ensure padding <= kernel_size // 2, a requirement for the underlying max_pool1d.
    def get_pool_outputs(original_shape, kernel_size, stride, padding):
        # This pooling operation must be valid to generate inputs for unpooling.
        # Handle cases where input is too small for the kernel, which is valid and results in 0-length output.
        if original_shape[-1] > 0 and original_shape[-1] + 2 * padding < kernel_size:
            original_tensor = torch.empty(original_shape, dtype=torch.float32)
        else:
            original_tensor = torch.randn(original_shape, dtype=torch.float32)

        pooled, indices = torch.nn.functional.max_pool1d(
            original_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True
        )
        return pooled.numpy(), indices.numpy(), original_tensor.shape

    # Case 1: Trivial case: kernel_size=1, stride=1 (identity-like)
    shape, ks, s, p = (1, 2, 5), 1, 1, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    # Case 2: Simplest non-trivial, non-overlapping case (stride=kernel_size)
    shape, ks, s, p = (1, 1, 4), 2, 2, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    # Case 3: Empty batch tensor. Should bypass main computation.
    shape, ks, s, p = (0, 3, 10), 2, 2, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))
    
    # Case 4: Non-overlapping with padding
    shape, ks, s, p = (1, 2, 8), 3, 3, 1 # padding=1 <= kernel_size//2=1
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    # Case 5: Gappy unpooling (stride > kernel_size)
    shape, ks, s, p = (2, 3, 10), 2, 3, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    # Case 6: Input resulting in an empty pooled tensor (length dim is 0)
    shape, ks, s, p = (1, 1, 1), 2, 2, 0 # L_in=1 < ks=2, results in 0-length output
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))
    
    # Case 7: 2D input, non-overlapping
    shape, ks, s, p = (4, 8), 2, 2, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))
    
    # Case 8: Large kernel, non-overlapping, with padding
    shape, ks, s, p = (1, 1, 25), 5, 5, 2 # padding=2 <= kernel_size//2=2
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    # Case 9: Another trivial case, kernel_size=1 with stride > 1
    shape, ks, s, p = (1, 4, 10), 1, 2, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    # Case 10: Input length is an exact multiple of stride/kernel
    shape, ks, s, p = (2, 3, 12), 4, 4, 0
    inp, ind, os = get_pool_outputs(shape, ks, s, p)
    list_of_inputs.append(copy.deepcopy({'input': inp, 'indices': ind, 'kernel_size': ks, 'stride': s, 'padding': p, 'output_size': os}))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_unpool1d"] = max_unpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_unpool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_unpool1d'.")

check_valid('torch.nn.functional.max_unpool1d', generated_inputs['torch.nn.functional.max_unpool1d'], lib="torch", suffix=0)

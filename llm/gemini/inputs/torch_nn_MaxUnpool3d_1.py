
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def get_maxunpool3d_inputs():
    list_of_inputs = []

    def _generate_input_deterministic(original_shape, kernel_size, stride, padding, dtype=np.float32):
        def _get_output_dim(in_dim, pad, ker, stri):
            # This formula is for MaxPool3d, which is what we are reversing
            return (in_dim + 2 * pad - ker) // stri + 1

        is_5d = len(original_shape) == 5
        if is_5d:
            N, C, D_in, H_in, W_in = original_shape
        else:
            C, D_in, H_in, W_in = original_shape
            N = 1

        D_out = _get_output_dim(D_in, padding, kernel_size, stride)
        H_out = _get_output_dim(H_in, padding, kernel_size, stride)
        W_out = _get_output_dim(W_in, padding, kernel_size, stride)

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            return None

        pooled_shape = (N, C, D_out, H_out, W_out) if is_5d else (C, D_out, H_out, W_out)
        
        # The 'input' to MaxUnpool3d is the output of MaxPool3d
        unpool_input_np = np.random.randn(*pooled_shape).astype(dtype)
        indices_np = np.zeros(pooled_shape, dtype=np.int64)
        num_spatial_elements = D_in * H_in * W_in

        for n in range(N):
            for c in range(C):
                num_indices_per_channel = D_out * H_out * W_out
                if num_indices_per_channel > num_spatial_elements:
                    return None # Cannot generate unique indices
                
                # Generate unique indices for this channel to ensure deterministic behavior
                flat_indices = np.random.choice(num_spatial_elements, size=num_indices_per_channel, replace=False)
                reshaped_indices = flat_indices.reshape((D_out, H_out, W_out))

                if is_5d:
                    indices_np[n, c] = reshaped_indices
                else:
                    indices_np[c] = reshaped_indices
    
        input_dict = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'input': unpool_input_np,
            'indices': indices_np,
            'output_size': original_shape
        }
        return input_dict

    params = [
        {'original_shape': (2, 3, 10, 10, 10), 'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'original_shape': (1, 1, 16, 16, 16), 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'original_shape': (3, 10, 10, 10), 'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'original_shape': (2, 2, 20, 20, 20), 'kernel_size': 5, 'stride': 3, 'padding': 2},
        {'original_shape': (1, 1, 5, 5, 5), 'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'original_shape': (1, 1, 8, 8, 8), 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dtype': np.float64},
        {'original_shape': (4, 9, 9, 9), 'kernel_size': 3, 'stride': 3, 'padding': 1},
        {'original_shape': (3, 2, 7, 8, 9), 'kernel_size': 2, 'stride': 1, 'padding': 0},
        {'original_shape': (1, 1, 2, 2, 2), 'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'original_shape': (1, 2, 10, 12, 14), 'kernel_size': 3, 'stride': 2, 'padding': 1},
    ]

    for p in params:
        dtype = p.get('dtype', np.float32)
        input_data = _generate_input_deterministic(p['original_shape'], p['kernel_size'], p['stride'], p['padding'], dtype)
        if input_data:
            list_of_inputs.append(copy.deepcopy(input_data))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool3d_1"] = get_maxunpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxUnpool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool3d_1'.")

check_valid('torch.nn.MaxUnpool3d', generated_inputs['torch.nn.MaxUnpool3d_1'], lib="torch", suffix=1)

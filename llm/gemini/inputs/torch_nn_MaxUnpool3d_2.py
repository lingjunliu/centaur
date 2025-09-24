
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def maxunpool3d_inputs():
    list_of_inputs = []

    # Helper function to generate inputs that are valid for deterministic execution.
    # The non-deterministic error is tied to the parallel implementation on CPU.
    # By setting batch size (N) and channels (C) to 1, we force a serial execution
    # path for the outer loop of the underlying kernel, which avoids the race condition
    # that makes the operation non-deterministic.
    # We also maintain other good practices:
    # 1. Non-overlapping windows (stride >= kernel_size) and unique input values
    #    (from torch.arange) to prevent duplicate max indices.
    # 2. Valid padding (pad <= kernel_size / 2).
    def generate_case(spatial_shape, kernel_size, stride, padding):
        # Enforce N=1, C=1 for deterministic behavior
        original_shape = (1, 1) + spatial_shape
        
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        st = stride if isinstance(stride, tuple) else (stride,) * 3
        pd = padding if isinstance(padding, tuple) else (padding,) * 3

        # Use arange to ensure every value is unique, so the max index is unambiguous.
        num_elements = np.prod(original_shape)
        original_input = torch.arange(num_elements, dtype=torch.float32).reshape(original_shape)
        
        pool = torch.nn.MaxPool3d(ks, st, pd, return_indices=True)
        pooled_output, indices = pool(original_input)
        
        input_dict = {
            'kernel_size': ks,
            'stride': st,
            'padding': pd,
            'input': pooled_output.numpy(),
            'indices': indices.numpy(),
            'output_size': tuple(original_shape)
        }
        return input_dict

    # All cases now have N=1, C=1 to ensure determinism.
    # Case 1: Basic case, stride=kernel, no padding.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(8, 10, 12),
        kernel_size=2, stride=2, padding=0)))

    # Case 2: Stride > kernel, with padding.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(9, 9, 13),
        kernel_size=3, stride=4, padding=1)))

    # Case 3: Non-cubic params, stride=kernel.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(8, 9, 12),
        kernel_size=(2, 3, 4), stride=(2, 3, 4), padding=0)))

    # Case 4: Non-cubic padding, stride > kernel.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(13, 13, 13),
        kernel_size=3, stride=4, padding=1)))

    # Case 5: Trivial kernel=1, stride > 1 (subsampling).
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(10, 13, 7),
        kernel_size=1, stride=3, padding=0)))

    # Case 6: Larger tensor, stride=kernel, with padding.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(18, 18, 18),
        kernel_size=4, stride=4, padding=1)))

    # Case 7: Minimal valid size.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(2, 2, 2),
        kernel_size=2, stride=2, padding=0)))

    # Case 8: Another non-cubic case, stride > kernel.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(9, 9, 9),
        kernel_size=(2, 2, 3), stride=(3, 3, 4), padding=(1, 1, 1))))

    # Case 9: Large stride with padding.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(20, 30, 40),
        kernel_size=2, stride=10, padding=1)))
        
    # Case 10: Another case with padding and stride=kernel.
    list_of_inputs.append(copy.deepcopy(generate_case(
        spatial_shape=(7, 7, 7),
        kernel_size=3, stride=3, padding=1)))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool3d_2"] = maxunpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxUnpool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool3d_2'.")

check_valid('torch.nn.MaxUnpool3d', generated_inputs['torch.nn.MaxUnpool3d_2'], lib="torch", suffix=2)

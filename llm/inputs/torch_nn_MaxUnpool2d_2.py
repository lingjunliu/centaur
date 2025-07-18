
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def maxunpool2d_inputs():
    list_of_inputs = []

    def _get_deterministic_input(input_tensor, kernel_size, stride):
        """
        Generates inputs for MaxUnpool2d that are guaranteed to trigger the
        deterministic CUDA kernel by using non-overlapping pooling windows
        (stride >= kernel_size) and zero padding.
        """
        padding_tup = (0, 0)
        kernel_size_tup = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        if stride is None:
            stride_tup = kernel_size_tup
        else:
            stride_tup = stride if isinstance(stride, tuple) else (stride, stride)

        # This check is crucial for determinism.
        if stride_tup[0] < kernel_size_tup[0] or stride_tup[1] < kernel_size_tup[1]:
            return None

        pool = nn.MaxPool2d(kernel_size, stride=stride, padding=0, return_indices=True)
        original_size = input_tensor.size()
        
        try:
            output, indices = pool(input_tensor.float())
        except Exception:
            return None

        return {
            'kernel_size': kernel_size_tup,
            'stride': stride_tup,
            'padding': padding_tup,
            'input': output.numpy(),
            'indices': indices.numpy(),
            'output_size': tuple(original_size)
        }

    # All cases use padding=0 and stride >= kernel_size to ensure determinism.

    # Case 1: Core case, simple tiling (stride == kernel_size)
    shape = (1, 1, 4, 4)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=2, stride=2))

    # Case 2: Multiple channels, tiling
    shape = (1, 3, 6, 8)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=2, stride=2))

    # Case 3: Multiple batches, tiling
    shape = (4, 1, 6, 9)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=3, stride=3))

    # Case 4: No batch dimension (C, H, W), tiling
    shape = (2, 10, 10)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=5, stride=5))

    # Case 5: Larger kernel, tiling
    shape = (1, 1, 16, 16)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=4, stride=4))

    # Case 6: Non-square kernel and stride (tiling)
    shape = (1, 2, 6, 8)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=(3, 2), stride=(3, 2)))

    # Case 7: Stride is None (defaults to kernel_size, which is a tiling case)
    shape = (1, 2, 9, 9)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=3, stride=None))

    # Case 8: Documentation example (which is deterministic: k=2, s=2, p=0)
    input_tensor = torch.tensor([[[[ 1., 2., 3., 4.], [ 5., 6., 7., 8.], [ 9., 10., 11., 12.], [13., 14., 15., 16.]]]])
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=2, stride=2))

    # Case 9: Stride > kernel_size (non-overlapping with gaps)
    shape = (2, 2, 8, 8)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=2, stride=4))
    
    # Case 10: Non-square tensor shape with tiling
    shape = (1, 1, 5, 10)
    input_tensor = torch.arange(float(np.prod(shape))).reshape(shape)
    list_of_inputs.append(_get_deterministic_input(input_tensor, kernel_size=(5, 5), stride=(5, 5)))

    final_list = [copy.deepcopy(item) for item in list_of_inputs if item is not None]
    
    return final_list

generated_inputs["torch.nn.MaxUnpool2d_2"] = maxunpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxUnpool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool2d_2'.")

check_valid('torch.nn.MaxUnpool2d', generated_inputs['torch.nn.MaxUnpool2d_2'], lib="torch", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    # The recurring error `does not have a deterministic implementation` is due to a
    # constraint in the testing environment (`torch.use_deterministic_algorithms(True)`),
    # which conflicts with the non-deterministic nature of the `max_unpool2d` CUDA kernel.
    # To maximize the chance of success, these inputs use `torch.arange` to ensure unique
    # values in the source tensor. This makes the `MaxPool2d` part of the operation
    # deterministic (as each `max` is unique), which is the best that can be done from the
    # input generation side. The inputs that previously caused `MaxPool2d` errors (e.g., with
    # zero-sized dimensions) have been removed.

    # Case 1: Minimal valid case
    original_shape = (1, 1, 2, 2)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic case with a larger tensor
    original_shape = (1, 1, 4, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 3: With batching
    original_shape = (2, 1, 4, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: With multiple channels
    original_shape = (1, 3, 4, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: With padding
    original_shape = (1, 1, 4, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (1, 1)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: With overlapping stride
    original_shape = (1, 1, 3, 3)
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Non-square kernel and stride (non-overlapping)
    original_shape = (1, 1, 4, 6)
    kernel_size = (2, 3)
    stride = (2, 3)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: float64 dtype
    original_shape = (1, 1, 2, 2)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float64).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Batch and channels combined
    original_shape = (2, 2, 4, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: Non-square padding
    original_shape = (1, 1, 3, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (1, 0)
    original_tensor = torch.arange(np.prod(original_shape), dtype=torch.float32).reshape(original_shape)
    pool = torch.nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
    pooled_output, indices = pool(original_tensor)
    input_dict = {
        'input': pooled_output.numpy(),
        'indices': indices.numpy(),
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'output_size': tuple(original_tensor.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_unpool2d"] = max_unpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_unpool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_unpool2d'.")

check_valid('torch.nn.functional.max_unpool2d', generated_inputs['torch.nn.functional.max_unpool2d'], lib="torch", suffix=0)

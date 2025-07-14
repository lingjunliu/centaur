
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def rand_like_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3], dtype=np.float32)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = True
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([1, 2, 3], dtype=np.float16)
    dtype = torch.float16
    layout = torch.strided
    requires_grad = True
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float64)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = False
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_tensor = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = True
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.zeros((2, 2, 2), dtype=np.float64)
    dtype = torch.float64
    layout = torch.strided
    requires_grad = False
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = np.ones((5), dtype=np.float32)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = True
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 3).astype(np.float32)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = False
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1, 2, 3], dtype=np.float16)
    dtype = torch.float16
    layout = torch.strided
    requires_grad = True
    memory_format = torch.contiguous_format
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.random.rand(1,3,2,2).astype(np.float32)
    dtype = torch.float32
    layout = torch.strided
    requires_grad = True
    memory_format = torch.channels_last
    input_dict = {"input": input_tensor, "dtype": dtype, "layout": layout, "requires_grad": requires_grad, "memory_format": memory_format}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.rand_like"] = rand_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.rand_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rand_like'.")

check_valid('torch.rand_like', generated_inputs['torch.rand_like'], lib="torch", suffix=0)

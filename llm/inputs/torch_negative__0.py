
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy
import copy

def negative__inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 tensor
    input_tensor = numpy.array([1.0, -2.5, 0.0, 3.14], dtype=numpy.float32)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 2: 2D int32 tensor
    input_tensor = numpy.array([[1, -2, 3], [-4, 5, -6]], dtype=numpy.int32)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 3: 0D (scalar) float64 tensor
    input_tensor = numpy.array(-42.42, dtype=numpy.float64)
    out_tensor = numpy.array(0.0, dtype=numpy.float64)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 4: 3D int8 tensor with positive and negative values
    input_tensor = numpy.arange(-12, 12, dtype=numpy.int8).reshape(2, 3, 4)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 5: Large int64 values
    input_tensor = numpy.array([9223372036854775807, -9223372036854775807, 0], dtype=numpy.int64)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))
    
    # Input 6: High-dimensional (4D) tensor with float16
    input_tensor = numpy.random.randn(2, 2, 3, 3).astype(numpy.float16)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 7: Integer boundary case for int8
    input_tensor = numpy.array([-128, -127, 0, 127], dtype=numpy.int8)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 8: Tensor with all zeros
    input_tensor = numpy.zeros((5, 5), dtype=numpy.float32)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 9: Long 1D tensor
    input_tensor = numpy.linspace(-100, 100, 200, dtype=numpy.float64)
    out_tensor = numpy.zeros_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    # Input 10: Uninitialized out tensor
    input_tensor = numpy.array([5, 10, 15], dtype=numpy.int16)
    out_tensor = numpy.empty_like(input_tensor)
    list_of_inputs.append(copy.deepcopy({'input': input_tensor, 'out': out_tensor}))

    return list_of_inputs

generated_inputs["torch.negative_"] = negative__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.negative_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.negative_'.")

check_valid('torch.negative_', generated_inputs['torch.negative_'], lib="torch", suffix=0)

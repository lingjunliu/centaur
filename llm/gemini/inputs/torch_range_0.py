
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy

def range_inputs():
    list_of_inputs = []

    # Case 1: Basic usage
    start, end, step = 1.0, 5.0, 1.0
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float step
    start, end, step = 0.0, 2.0, 0.5
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative start and end
    start, end, step = -10.0, -5.0, 1.0
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative step (descending order)
    start, end, step = 5.0, -5.0, -2.5
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: With specified dtype (float64)
    start, end, step = 0.0, 10.0, 2.0
    dtype = numpy.float64
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: With requires_grad=True
    start, end, step = 0.0, 3.0, 1.0
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Single element output (start == end)
    start, end, step = 7.7, 7.7, 1.0
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: All arguments provided with different dtype and requires_grad
    start, end, step = 8.0, -8.0, -4.0
    dtype = numpy.float64
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: With specified dtype (float16)
    start, end, step = 0.0, 1.0, 0.1
    dtype = numpy.float16
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 10: Using default start and step values explicitly
    start, end, step = 0.0, 4.0, 1.0
    dtype = numpy.float32
    size = int(numpy.floor((end - start) / step) + 1)
    input_dict = {
        'start': start, 'end': end, 'step': step,
        'out': numpy.zeros(size, dtype=dtype), 'dtype': dtype, 'layout': 'torch.strided', 'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.range"] = range_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.range'.")

check_valid('torch.range', generated_inputs['torch.range'], lib="torch", suffix=0)

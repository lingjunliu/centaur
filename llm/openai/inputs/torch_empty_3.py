
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_empty_inputs():
    list_of_inputs = []

    size = [2, 3]
    out = np.empty((2, 3), dtype=np.dtype('float32'))
    dtype = np.dtype('float32')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = []
    out = np.empty((), dtype=np.dtype('float64'))
    dtype = np.dtype('float64')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [0]
    out = np.empty((0,), dtype=np.dtype('int64'))
    dtype = np.dtype('int64')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [4, 0, 5]
    out = np.empty((4, 0, 5), dtype=np.dtype('float16'))
    dtype = np.dtype('float16')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [1, 2, 3, 4]
    out = np.empty((1, 2, 3, 4), dtype=np.dtype('complex64'))
    dtype = np.dtype('complex64')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [10]
    out = np.empty((10,), dtype=np.dtype('uint8'))
    dtype = np.dtype('uint8')
    requires_grad = False
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [3, 3]
    out = np.empty((3, 3), dtype=np.dtype('bool'))
    dtype = np.dtype('bool')
    requires_grad = False
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [5]
    out = np.empty((5,), dtype=np.dtype('int32'))
    dtype = np.dtype('int32')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [2, 2, 2]
    out = np.empty((2, 2, 2), dtype=np.dtype('float64'))
    dtype = np.dtype('float64')
    requires_grad = True
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [7]
    out = np.empty((7,), dtype=np.dtype('complex128'))
    dtype = np.dtype('complex128')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [8, 1]
    out = np.empty((8, 1), dtype=np.dtype('float32'))
    dtype = np.dtype('float32')
    requires_grad = False
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    size = [2, 3, 4, 5, 6]
    out = np.empty((2, 3, 4, 5, 6), dtype=np.dtype('int8'))
    dtype = np.dtype('int8')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({
        "size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory
    }))

    return list_of_inputs

generated_inputs["torch.empty_3"] = torch_empty_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_3'.")


check_valid('torch.empty', generated_inputs['torch.empty_3'], lib="torch", suffix=3)

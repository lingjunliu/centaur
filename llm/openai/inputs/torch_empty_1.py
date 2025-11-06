
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def empty_inputs():
    list_of_inputs = []

    # 1
    size = (0,)
    out = np.empty(size, dtype=np.float32)
    dtype = np.dtype('float32')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 2
    size = (1,)
    out = np.zeros(size, dtype=np.int64)
    dtype = np.dtype('int64')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 3
    size = (5,)
    out = np.empty(size, dtype=np.float64)
    dtype = np.dtype('float64')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 4
    size = (10,)
    out = np.ones(size, dtype=np.int32)
    dtype = np.dtype('int32')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 5
    size = (3,)
    out = np.zeros(size, dtype=np.complex64)
    dtype = np.dtype('complex64')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 6
    size = (8,)
    out = np.zeros(size, dtype=np.float16)
    dtype = np.dtype('float16')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 7
    size = (2,)
    out = np.zeros(size, dtype=np.bool_)
    dtype = np.dtype('bool')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 8
    size = (4,)
    out = np.arange(4, dtype=np.uint8)
    dtype = np.dtype('uint8')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 9
    size = (6,)
    out = np.zeros(size, dtype=np.complex128)
    dtype = np.dtype('complex128')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 10
    size = (7,)
    out = np.linspace(0, 1, 7, dtype=np.float32)
    dtype = np.dtype('float32')
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 11
    size = (12,)
    out = np.zeros(size, dtype=np.int8)
    dtype = np.dtype('int8')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    # 12
    size = (9,)
    out = np.full(size, 3.14, dtype=np.float64)
    dtype = np.dtype('float64')
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    return list_of_inputs

generated_inputs["torch.empty_1"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_1'.")


check_valid('torch.empty', generated_inputs['torch.empty_1'], lib="torch", suffix=1)

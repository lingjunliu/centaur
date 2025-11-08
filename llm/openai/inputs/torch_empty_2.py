
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def empty_inputs():
    list_of_inputs = []

    size = (3,)
    out = np.empty(size, dtype=np.float32)
    dtype = np.dtype(np.float32)
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (2, 3)
    out = np.empty(size, dtype=np.float64)
    dtype = np.dtype(np.float64)
    requires_grad = True
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (0,)
    out = np.empty(size, dtype=np.int64)
    dtype = np.dtype(np.int64)
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (2, 0)
    out = np.empty(size, dtype=np.int32)
    dtype = np.dtype(np.int32)
    requires_grad = False
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = ()
    out = np.empty(size, dtype=np.float16)
    dtype = np.dtype(np.float16)
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (1, 2, 3)
    out = np.empty(size, dtype=np.complex64)
    dtype = np.dtype(np.complex64)
    requires_grad = True
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (4, 1, 2, 1)
    out = np.empty(size, dtype=np.bool_)
    dtype = np.dtype(np.bool_)
    requires_grad = False
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (2, 2, 2, 2, 2)
    out = np.empty(size, dtype=np.int16)
    dtype = np.dtype(np.int16)
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (1, 1, 1, 1, 1, 1, 1, 1)
    out = np.empty(size, dtype=np.complex128)
    dtype = np.dtype(np.complex128)
    requires_grad = True
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (3, 4, 5)
    out = np.empty(size, dtype=np.uint8)
    dtype = np.dtype(np.uint8)
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (2, 3, 0, 4)
    out = np.empty(size, dtype=np.float32)
    dtype = np.dtype(np.float32)
    requires_grad = True
    pin_memory = True
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    size = (7,)
    out = np.empty(size, dtype=np.int8)
    dtype = np.dtype(np.int8)
    requires_grad = False
    pin_memory = False
    list_of_inputs.append(copy.deepcopy({"size": size, "out": out, "dtype": dtype, "requires_grad": requires_grad, "pin_memory": pin_memory}))

    return list_of_inputs

generated_inputs["torch.empty_2"] = empty_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_2'.")


check_valid('torch.empty', generated_inputs['torch.empty_2'], lib="torch", suffix=2)

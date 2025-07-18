
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy

def sym_fresh_size_inputs():
    list_of_inputs = []
    # The error "TypeError: sym_fresh_size() missing 1 required positional argument: 'expr'"
    # has been persistent across multiple attempts with different types passed as a keyword argument.
    # This suggests two possibilities:
    # 1. The argument is positional-only. The error message "missing 1 required positional argument"
    #    is a strong indicator for this.
    # 2. The type of the argument is a special type (`SymInt`) which cannot be easily created,
    #    and the dispatcher fails to find any matching overload, resulting in a generic error.
    #
    # This attempt will address the first possibility by passing the argument positionally.
    # The testing framework seems to support this via an "args" key in the input dictionary.
    # We will provide a variety of plausible scalar types positionally, adhering to the
    # "numpy format" rule where possible.

    # Inputs with numpy scalar integers (positional)
    input_dict_1 = {"args": [numpy.int64(0)], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {"args": [numpy.int64(1)], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {"args": [numpy.int32(10)], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Inputs with 0-dim numpy arrays (representing tensors) (positional)
    input_dict_4 = {"args": [numpy.array(5, dtype=numpy.int64)], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {"args": [numpy.array(100, dtype=numpy.int32)], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    input_dict_6 = {"args": [numpy.array(2, dtype=numpy.int64)], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Inputs with primitive Python integers (positional)
    input_dict_7 = {"args": [8], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_dict_8 = {"args": [16], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_dict_9 = {"args": [32], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_dict_10 = {"args": [64], "kwargs": {}}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.sym_fresh_size"] = sym_fresh_size_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sym_fresh_size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_fresh_size'.")

check_valid('torch.sym_fresh_size', generated_inputs['torch.sym_fresh_size'], lib="torch", suffix=0)

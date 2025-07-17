
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def einsum_inputs():
    list_of_inputs = []

    # Input 1: Matrix multiplication
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 4)
    operands = [A, B]
    input_dict = {"operands": operands}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Trace of a matrix
    A = np.random.rand(5, 5)
    operands = [A]
    input_dict = {"operands": operands}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.einsum_2"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.einsum_2'.")

check_valid('torch.einsum', generated_inputs['torch.einsum_2'], lib="torch", suffix=2)

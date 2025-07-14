
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def multigammaln_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    p = 2
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    p = 1
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    p = 3
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    p = 2
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([5.0], dtype=np.float64)
    p = 4
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([2.718, 3.141, 1.618], dtype=np.float64)
    p = 1
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
    p = 3
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    p = 1
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.array([[[10.0, 11.0], [12.0, 13.0]], [[14.0, 15.0], [16.0, 17.0]]], dtype=np.float64)
    p = 1
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    p = 5
    input_dict = {"input": input, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.multigammaln"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.multigammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.multigammaln'.")

check_valid('torch.special.multigammaln', generated_inputs['torch.special.multigammaln'], lib="torch", suffix=0)

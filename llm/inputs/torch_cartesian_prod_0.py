
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def cartesian_prod_inputs():
    list_of_inputs = []

    def create_input(arr_list):
        return {"tensors": arr_list}

    # Input 1: Basic 1D tensors
    inputs = create_input([np.array([1, 2]), np.array([3, 4])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 5: tensors with different data types
    inputs = create_input([np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int64)])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 7: Tensors with different sizes
    inputs = create_input([np.array([1, 2, 3]), np.array([4])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 9: More tensors
    inputs = create_input([np.array([1]), np.array([2]), np.array([3]), np.array([4])])
    list_of_inputs.append(copy.deepcopy(inputs))
    
    # Input 10: float tensors
    inputs = create_input([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 11: Single tensor
    inputs = create_input([np.array([1, 2, 3])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 12: More different sizes
    inputs = create_input([np.array([1]), np.array([2,3])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 13: Negative values
    inputs = create_input([np.array([-1, 2]), np.array([3, -4])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 14: Zero values
    inputs = create_input([np.array([0, 2]), np.array([3, 0])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 15: boolean values
    inputs = create_input([np.array([True, False]), np.array([True, True])])
    list_of_inputs.append(copy.deepcopy(inputs))

    # Input 16: Different data types combined
    inputs = create_input([np.array([1, 2], dtype=np.int32), np.array([3.0, 4.0])])
    list_of_inputs.append(copy.deepcopy(inputs))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cartesian_prod"] = cartesian_prod_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cartesian_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cartesian_prod'.")

check_valid('torch.cartesian_prod', generated_inputs['torch.cartesian_prod'], lib="torch", suffix=0)

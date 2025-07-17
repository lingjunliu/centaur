
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def cartesian_prod_inputs():
    list_of_inputs = []

    # Input 1: Two 1D tensors
    tensors = [np.array([1, 2]), np.array([3, 4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two 1D tensors with different dtypes
    tensors = [np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64)]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three 1D tensors
    tensors = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One tensor
    tensors = [np.array([1, 2, 3])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Three tensors with different sizes
    tensors = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two 1D tensors with negative values
    tensors = [np.array([-1, 2]), np.array([3, -4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Several tensors of different dimensionalities and sizes. Reduced dimensions and removed the multidimensional one
    tensors = [np.array([1, 2]), np.array([3, 4]), np.array([9])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All tensors 1D
    tensors = [np.array([1, 2]), np.array([3,4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Add a small float and int tensor
    tensors = [np.array([1.0, 2.0]), np.array([3, 4])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Large tensors
    tensors = [np.array(np.arange(10)), np.array(np.arange(5))]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

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

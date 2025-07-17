
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def from_numpy_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    ndarray = np.array([1, 2, 3, 4, 5])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of floats
    ndarray = np.array([[1.1, 2.2], [3.3, 4.4]])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array of complex numbers
    ndarray = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty array
    ndarray = np.array([])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array with negative values
    ndarray = np.array([-1, -2, -3])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with a specific dtype (int64)
    ndarray = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with a specific dtype (float32)
    ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multidimensional array
    ndarray = np.random.rand(2, 3, 4)
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with zeros
    ndarray = np.zeros((2, 2))
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with ones
    ndarray = np.ones((3, 3))
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.from_numpy"] = from_numpy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.from_numpy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_numpy'.")

check_valid('torch.from_numpy', generated_inputs['torch.from_numpy'], lib="torch", suffix=0)

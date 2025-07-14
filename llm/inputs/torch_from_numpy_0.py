
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

    # Input 3: 3D array of booleans
    ndarray = np.array([[[True, False], [False, True]], [[True, True], [False, False]]])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of negative integers
    ndarray = np.array([-1, -2, -3, -4, -5])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of mixed positive and negative floats
    ndarray = np.array([[-1.1, 2.2], [-3.3, 4.4]])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty array
    ndarray = np.array([])
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional array (4D) - removing to reduce possible errors.
    # ndarray = np.random.rand(2, 3, 4, 5)
    # input_dict = {"ndarray": ndarray}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with a specific dtype (int64)
    ndarray = np.array([10, 20, 30], dtype=np.int64)
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with a specific dtype (float32)
    ndarray = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"ndarray": ndarray}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero dimensional array
    ndarray = np.array(10)
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

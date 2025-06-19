
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ShortStorage_inputs():
    list_of_inputs = []

    # Input 1: Empty storage, create a numpy array

    numpy_array_empty = np.array([], dtype=np.int16)
    input_dict = {
        "storage": numpy_array_empty
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Storage from a list, convert to numpy array first

    numpy_array_list = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    input_dict = {
        "storage": numpy_array_list
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Storage from a tuple, convert to numpy array first

    numpy_array_tuple = np.array([6, 7, 8, 9, 10], dtype=np.int16)
    input_dict = {
        "storage": numpy_array_tuple
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Storage from a numpy array

    numpy_array = np.array([11, 12, 13, 14, 15], dtype=np.int16)
    input_dict = {
        "storage": numpy_array
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Storage with negative values

    numpy_array_negative = np.array([-1, -2, -3, -4, -5], dtype=np.int16)
    input_dict = {
        "storage": numpy_array_negative
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.ShortStorage_4"] = ShortStorage_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.ShortStorage_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ShortStorage_4'.")

check_valid('torch.ShortStorage', generated_inputs['torch.ShortStorage_4'], lib="torch", suffix=4)

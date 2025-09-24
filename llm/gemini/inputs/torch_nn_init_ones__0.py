
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ones_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    tensor = np.zeros(5, dtype=np.float32)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor
    tensor = np.zeros((3, 4), dtype=np.float64)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    tensor = np.zeros((2, 3, 5), dtype=np.int32)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: tensor with different data type
    tensor = np.zeros((2, 2), dtype=np.complex64)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: empty tensor
    tensor = np.array([])
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large tensor
    tensor = np.zeros((100, 100), dtype=np.float32)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor
    tensor = np.zeros((2, 3, 4, 2), dtype=np.float32)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Bool tensor
    tensor = np.zeros((2,2), dtype=bool)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int tensor
    tensor = np.zeros(5, dtype=np.int64)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor complex
    tensor = np.zeros((2,2,2), dtype=np.complex128)
    input_dict = {"tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.init.ones_"] = ones_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.init.ones_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.ones_'.")

check_valid('torch.nn.init.ones_', generated_inputs['torch.nn.init.ones_'], lib="torch", suffix=0)

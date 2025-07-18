
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def dirac__inputs():
    list_of_inputs = []

    # Input 1: Basic 3D tensor, offset=1 (equivalent to default groups=1)
    tensor1 = np.zeros((4, 2, 5), dtype=np.float32)
    input_dict1 = {
        'tensor': tensor1,
        'offset': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic 4D tensor, offset=1
    tensor2 = np.zeros((6, 3, 3, 3), dtype=np.float32)
    input_dict2 = {
        'tensor': tensor2,
        'offset': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Basic 5D tensor, offset=1
    tensor3 = np.zeros((8, 4, 3, 3, 3), dtype=np.float32)
    input_dict3 = {
        'tensor': tensor3,
        'offset': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor with a positive offset > 1
    tensor4 = np.zeros((10, 5, 5, 5), dtype=np.float32)
    input_dict4 = {
        'tensor': tensor4,
        'offset': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor with a different positive offset
    tensor5 = np.zeros((9, 3, 7, 7), dtype=np.float32)
    input_dict5 = {
        'tensor': tensor5,
        'offset': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D tensor with a positive offset
    tensor6 = np.zeros((12, 6, 9), dtype=np.float32)
    input_dict6 = {
        'tensor': tensor6,
        'offset': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 5D tensor with a positive offset where offset divides the first dim
    tensor7 = np.zeros((4, 2, 5, 5, 5), dtype=np.float32)
    input_dict7 = {
        'tensor': tensor7,
        'offset': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 4D tensor with float64 dtype and a large offset
    tensor8 = np.zeros((16, 8, 3, 3), dtype=np.float64)
    input_dict8 = {
        'tensor': tensor8,
        'offset': 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 4D tensor with non-square kernel
    tensor9 = np.zeros((8, 2, 3, 5), dtype=np.float32)
    input_dict9 = {
        'tensor': tensor9,
        'offset': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: 4D tensor with even-sized kernel and offset equal to the first dim
    tensor10 = np.zeros((6, 6, 4, 4), dtype=np.float32)
    input_dict10 = {
        'tensor': tensor10,
        'offset': 6
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.init.dirac_"] = dirac__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.dirac_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.dirac_'.")

check_valid('torch.nn.init.dirac_', generated_inputs['torch.nn.init.dirac_'], lib="torch", suffix=0)

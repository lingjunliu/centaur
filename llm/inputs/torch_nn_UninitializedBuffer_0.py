
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def uninitializedbuffer_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "size": (2, 3),
        "dtype": np.float32,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "size": (5,),
        "dtype": np.int64,
        "requires_grad": True,
        "pin_memory": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "size": (1, 4, 2),
        "dtype": np.float64,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "size": (10, 10),
        "dtype": np.uint8,
        "requires_grad": True,
        "pin_memory": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "size": (3, 3, 3),
        "dtype": np.int32,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "size": (2, 2, 2, 2),
        "dtype": np.float16,
        "requires_grad": True,
        "pin_memory": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "size": (7,),
        "dtype": np.bool_,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "size": (1, 5, 1),
        "dtype": np.int8,
        "requires_grad": True,
        "pin_memory": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "size": (4, 1, 4, 1),
        "dtype": np.complex64,
        "requires_grad": False,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "size": (2,),
        "dtype": np.complex128,
        "requires_grad": True,
        "pin_memory": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.UninitializedBuffer"] = uninitializedbuffer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.UninitializedBuffer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.UninitializedBuffer'.")

check_valid('torch.nn.UninitializedBuffer', generated_inputs['torch.nn.UninitializedBuffer'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def uninitializedbuffer_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 buffer
    input_dict_1 = {
        'size': (10,),
        'dtype': np.float32,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int64 buffer
    input_dict_2 = {
        'size': (5, 5),
        'dtype': np.int64,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D float64 buffer with requires_grad=True
    input_dict_3 = {
        'size': (2, 3, 4),
        'dtype': np.float64,
        'layout': 'strided',
        'requires_grad': True,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D boolean buffer with pin_memory=True
    input_dict_4 = {
        'size': (100,),
        'dtype': np.bool_,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar (0-dimensional) buffer
    input_dict_5 = {
        'size': (),
        'dtype': np.float32,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large 1D int32 buffer
    input_dict_6 = {
        'size': (10000,),
        'dtype': np.int32,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 2D complex64 buffer
    input_dict_7 = {
        'size': (8, 8),
        'dtype': np.complex64,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D complex128 buffer with pin_memory=True and requires_grad=True
    input_dict_8 = {
        'size': (4, 2),
        'dtype': np.complex128,
        'layout': 'strided',
        'requires_grad': True,
        'pin_memory': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: All flags True (with float16)
    input_dict_9 = {
        'size': (16,),
        'dtype': np.float16,
        'layout': 'strided',
        'requires_grad': True,
        'pin_memory': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Buffer with a zero dimension
    input_dict_10 = {
        'size': (5, 0, 5),
        'dtype': np.float32,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: 1D int8 buffer
    input_dict_11 = {
        'size': (128,),
        'dtype': np.int8,
        'layout': 'strided',
        'requires_grad': False,
        'pin_memory': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["torch.nn.UninitializedBuffer"] = uninitializedbuffer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.UninitializedBuffer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.UninitializedBuffer'.")

check_valid('torch.nn.UninitializedBuffer', generated_inputs['torch.nn.UninitializedBuffer'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy

def empty_strided_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D contiguous float tensor
    input_dict_1 = {
        'size': (2, 3),
        'stride': (3, 1),
        'dtype': torch.float32,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float tensor with requires_grad
    input_dict_2 = {
        'size': (5,),
        'stride': (1,),
        'dtype': torch.float64,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D non-contiguous (transposed view) integer tensor
    input_dict_3 = {
        'size': (3, 2),
        'stride': (1, 3),
        'dtype': torch.int32,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Pinned memory and gapped (non-contiguous) stride
    input_dict_4 = {
        'size': (10,),
        'stride': (2,),
        'dtype': torch.float32,
        'layout': torch.strided,
        'pin_memory': True,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Tensor with a zero-sized dimension
    input_dict_5 = {
        'size': (5, 0, 5),
        'stride': (0, 5, 1),
        'dtype': torch.uint8,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 0D tensor (scalar) with boolean type
    input_dict_6 = {
        'size': (),
        'stride': (),
        'dtype': torch.bool,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex type with requires_grad
    input_dict_7 = {
        'size': (2, 2),
        'stride': (2, 1),
        'dtype': torch.complex64,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Overlapping strides
    input_dict_8 = {
        'size': (4, 4),
        'stride': (1, 1),
        'dtype': torch.int64,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High-dimensional contiguous tensor with float16
    input_dict_9 = {
        'size': (2, 3, 2, 4),
        'stride': (24, 8, 4, 1),
        'dtype': torch.float16,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Broadcasted dimension (stride=0)
    input_dict_10 = {
        'size': (3, 5),
        'stride': (0, 1),
        'dtype': torch.int16,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Completely empty tensor (size=0)
    input_dict_11 = {
        'size': (0,),
        'stride': (1,),
        'dtype': torch.float32,
        'layout': torch.strided,
        'pin_memory': False,
        'requires_grad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    return list_of_inputs

generated_inputs["torch.empty_strided"] = empty_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_strided'.")

check_valid('torch.empty_strided', generated_inputs['torch.empty_strided'], lib="torch", suffix=0)

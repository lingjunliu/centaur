
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy

def fractional_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic 4D input with output_size
    input_dict_1 = {
        "input": torch.randn(1, 1, 10, 10).numpy(),
        "kernel_size": (2, 2),
        "output_size": (5, 5),
        "return_indices": False,
    }
    # In the actual call, `output_ratio` will be omitted if `output_size` is present.
    input_dict_1_final = copy.deepcopy(input_dict_1)
    input_dict_1_final['output_ratio'] = None # Placeholder for signature matching
    list_of_inputs.append(input_dict_1_final)


    # Input 2: Basic 4D input with output_ratio
    input_dict_2 = {
        "input": torch.randn(2, 3, 12, 12).numpy(),
        "kernel_size": (3, 3),
        "output_ratio": (0.5, 0.5),
        "return_indices": False,
    }
    input_dict_2_final = copy.deepcopy(input_dict_2)
    input_dict_2_final['output_size'] = None # Placeholder for signature matching
    list_of_inputs.append(input_dict_2_final)

    # Input 3: Basic 3D input with output_size
    input_dict_3 = {
        "input": torch.randn(4, 15, 15).numpy(),
        "kernel_size": (2, 3),
        "output_size": (7, 8),
        "return_indices": False,
    }
    input_dict_3_final = copy.deepcopy(input_dict_3)
    input_dict_3_final['output_ratio'] = None
    list_of_inputs.append(input_dict_3_final)


    # Input 4: Basic 3D input with output_ratio and return_indices=True
    input_dict_4 = {
        "input": torch.randn(2, 20, 20).numpy(),
        "kernel_size": (4, 4),
        "output_ratio": (0.25, 0.75),
        "return_indices": True,
    }
    input_dict_4_final = copy.deepcopy(input_dict_4)
    input_dict_4_final['output_size'] = None
    list_of_inputs.append(input_dict_4_final)


    # Input 5: 4D input with output_size and return_indices=True
    input_dict_5 = {
        "input": torch.randn(1, 1, 8, 8).numpy(),
        "kernel_size": (2, 2),
        "output_size": (4, 4),
        "return_indices": True,
    }
    input_dict_5_final = copy.deepcopy(input_dict_5)
    input_dict_5_final['output_ratio'] = None
    list_of_inputs.append(input_dict_5_final)


    # Input 6: 4D input with output_ratio and return_indices=True
    input_dict_6 = {
        "input": torch.randn(1, 1, 9, 9).numpy(),
        "kernel_size": (3, 3),
        "output_ratio": (0.6, 0.6),
        "return_indices": True,
    }
    input_dict_6_final = copy.deepcopy(input_dict_6)
    input_dict_6_final['output_size'] = None
    list_of_inputs.append(input_dict_6_final)


    # Input 7: Non-square kernel and output size
    input_dict_7 = {
        "input": torch.randn(1, 2, 16, 12).numpy(),
        "kernel_size": (3, 2),
        "output_size": (8, 7),
        "return_indices": False,
    }
    input_dict_7_final = copy.deepcopy(input_dict_7)
    input_dict_7_final['output_ratio'] = None
    list_of_inputs.append(input_dict_7_final)


    # Input 8: Non-square output ratio
    input_dict_8 = {
        "input": torch.randn(1, 1, 10, 20).numpy(),
        "kernel_size": (2, 2),
        "output_ratio": (0.7, 0.4),
        "return_indices": False,
    }
    input_dict_8_final = copy.deepcopy(input_dict_8)
    input_dict_8_final['output_size'] = None
    list_of_inputs.append(input_dict_8_final)


    # Input 9: Float64 dtype input
    input_dict_9 = {
        "input": torch.randn(1, 1, 7, 7, dtype=torch.float64).numpy(),
        "kernel_size": (2, 2),
        "output_size": (3, 3),
        "return_indices": False,
    }
    input_dict_9_final = copy.deepcopy(input_dict_9)
    input_dict_9_final['output_ratio'] = None
    list_of_inputs.append(input_dict_9_final)


    # Input 10: Input with negative values
    input_dict_10 = {
        "input": torch.randint(-10, 10, (1, 1, 10, 10)).float().numpy(),
        "kernel_size": (2, 2),
        "output_size": (6, 6),
        "return_indices": True,
    }
    input_dict_10_final = copy.deepcopy(input_dict_10)
    input_dict_10_final['output_ratio'] = None
    list_of_inputs.append(input_dict_10_final)
    
    # Clean up final list to match the provided signature strictly
    final_list = []
    for d in list_of_inputs:
        new_d = {
            'input': d['input'],
            'kernel_size': d['kernel_size'],
            'return_indices': d['return_indices'],
            'output_size': (),
            'output_ratio': ()
        }
        if d.get('output_size') is not None:
            new_d['output_size'] = d['output_size']
        if d.get('output_ratio') is not None:
            new_d['output_ratio'] = d['output_ratio']
        final_list.append(new_d)


    return final_list

generated_inputs["torch.nn.functional.fractional_max_pool2d"] = fractional_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.fractional_max_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.fractional_max_pool2d'.")

check_valid('torch.nn.functional.fractional_max_pool2d', generated_inputs['torch.nn.functional.fractional_max_pool2d'], lib="torch", suffix=0)

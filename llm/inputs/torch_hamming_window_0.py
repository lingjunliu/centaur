
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy
import copy

def hamming_window_inputs():
    """
    Generates a list of valid inputs for the torch.hamming_window function.
    """
    list_of_inputs = []
    
    # Base dictionary containing all keys from the signature with default values.
    # This is to avoid KeyError from the testing framework.
    # Crucially, use torch types for dtype and layout as specified in the error's expected signatures.
    base_dict = {
        'window_length': 50,
        'periodic': True,
        'alpha': 0.54,
        'beta': 0.46,
        'dtype': torch.float32,
        'layout': torch.strided,
        'requires_grad': False,
    }

    # Input 1: Basic case with default parameters
    input_dict_1 = base_dict.copy()
    input_dict_1['window_length'] = 10
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Non-periodic window
    input_dict_2 = base_dict.copy()
    input_dict_2['window_length'] = 21
    input_dict_2['periodic'] = False
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Different data type (float64)
    input_dict_3 = base_dict.copy()
    input_dict_3['window_length'] = 50
    input_dict_3['dtype'] = torch.float64
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Requires gradient
    input_dict_4 = base_dict.copy()
    input_dict_4['window_length'] = 8
    input_dict_4['requires_grad'] = True
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Custom alpha and beta (creates a Hann window)
    input_dict_5 = base_dict.copy()
    input_dict_5['window_length'] = 32
    input_dict_5['periodic'] = False
    input_dict_5['alpha'] = 0.5
    input_dict_5['beta'] = 0.5
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large window with float64 and requires_grad
    input_dict_6 = base_dict.copy()
    input_dict_6['window_length'] = 1024
    input_dict_6['dtype'] = torch.float64
    input_dict_6['requires_grad'] = True
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Edge case - window_length = 1 (periodic)
    input_dict_7 = base_dict.copy()
    input_dict_7['window_length'] = 1
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Edge case - window_length = 0 (returns an empty tensor)
    input_dict_8 = base_dict.copy()
    input_dict_8['window_length'] = 0
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Edge case - window_length = 1 (non-periodic)
    input_dict_9 = base_dict.copy()
    input_dict_9['window_length'] = 1
    input_dict_9['periodic'] = False
    input_dict_9['dtype'] = torch.float64
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Full combination of non-default parameters
    input_dict_10 = {
        'window_length': 100,
        'periodic': False,
        'alpha': 0.6,
        'beta': 0.4,
        'dtype': torch.float64,
        'layout': torch.strided,
        'requires_grad': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.hamming_window"] = hamming_window_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hamming_window' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hamming_window'.")

check_valid('torch.hamming_window', generated_inputs['torch.hamming_window'], lib="torch", suffix=0)

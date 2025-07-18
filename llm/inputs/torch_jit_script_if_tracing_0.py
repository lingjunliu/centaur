
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def script_if_tracing_inputs():
    list_of_inputs = []

    # The recurring error "TypeError: script_if_tracing() takes 1 positional argument but 2 were given"
    # arises from a conflict between the testing harness's requirements:
    # 1. The signature validation requires both 'fn' and 'alternative_fn' keys to be present in the input dictionary.
    # 2. The API runner then incorrectly calls the function with two positional arguments, causing the TypeError.
    # The actual `torch.jit.script_if_tracing` function can accept two arguments, but the specific error message
    # suggests a problem in how the test harness invokes the function.
    # To satisfy the signature validation and avoid the prior KeyError, both keys must be provided.
    # The following inputs strictly adhere to the user-provided signature `{'fn': 'list', 'alternative_fn': 'list'}`.

    # Input 1: Simple lists with 1D integer arrays
    input_dict_1 = {
        'fn': [np.array([1, 2, 3], dtype=np.int32)],
        'alternative_fn': [np.array([4, 5, 6], dtype=np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Simple lists with 1D float arrays
    input_dict_2 = {
        'fn': [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        'alternative_fn': [np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Simple lists with 2D integer arrays
    input_dict_3 = {
        'fn': [np.array([[1, 2], [3, 4]], dtype=np.int64)],
        'alternative_fn': [np.array([[5, 6], [7, 8]], dtype=np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty lists for both arguments
    input_dict_4 = {
        'fn': [],
        'alternative_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: One list empty, the other populated
    input_dict_5 = {
        'fn': [np.array([10, 20], dtype=np.uint8)],
        'alternative_fn': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Populated list for 'fn', empty for 'alternative_fn'
    input_dict_6 = {
        'fn': [],
        'alternative_fn': [np.array([True, False], dtype=np.bool_)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Lists with negative values
    input_dict_7 = {
        'fn': [np.array([-1, -5, -10])],
        'alternative_fn': [np.array([[-100], [-200]])]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Lists with different shapes
    input_dict_8 = {
        'fn': [np.ones((3, 1), dtype=np.float16)],
        'alternative_fn': [np.zeros((1, 3), dtype=np.float16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Lists with multiple arrays inside
    input_dict_9 = {
        'fn': [np.array([1]), np.array([2, 3])],
        'alternative_fn': [np.array([4, 5, 6]), np.array([7])]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Lists with higher-dimensional arrays
    input_dict_10 = {
        'fn': [np.zeros((2, 2, 2), dtype=np.int8)],
        'alternative_fn': [np.ones((1, 1, 1), dtype=np.int8)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.jit.script_if_tracing"] = script_if_tracing_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.script_if_tracing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.script_if_tracing'.")

check_valid('torch.jit.script_if_tracing', generated_inputs['torch.jit.script_if_tracing'], lib="torch", suffix=0)

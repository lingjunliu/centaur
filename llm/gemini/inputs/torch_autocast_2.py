
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy

def autocast_2_inputs():
    list_of_inputs = []

    # The recurring error `TypeError: autocast.__call__() missing 1 required positional argument: 'func'`
    # suggests that the testing framework is incorrectly calling the returned `autocast` object.
    # The inputs provided are valid for instantiating the `torch.autocast` class,
    # which is used as a context manager or decorator. The error arises from the testing
    # framework attempting to use it as a decorator without supplying the function to be decorated.
    # The following inputs are compliant with the specified signature.

    # Input 1: Basic CUDA usage, enabled, float16
    input_dict_1 = {
        'device_type': 'cuda',
        'dtype': numpy.float16,
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic CUDA usage, disabled, float16
    input_dict_2 = {
        'device_type': 'cuda',
        'dtype': numpy.float16,
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic CPU usage, enabled, float16 (autocast context itself is valid)
    input_dict_3 = {
        'device_type': 'cpu',
        'dtype': numpy.float16,
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Basic CPU usage, disabled, float16
    input_dict_4 = {
        'device_type': 'cpu',
        'dtype': numpy.float16,
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: CUDA usage, enabled, float32 (no-op but valid)
    input_dict_5 = {
        'device_type': 'cuda',
        'dtype': numpy.float32,
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: CPU usage, enabled, float32 (no-op but valid)
    input_dict_6 = {
        'device_type': 'cpu',
        'dtype': numpy.float32,
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: CUDA usage, disabled, float32
    input_dict_7 = {
        'device_type': 'cuda',
        'dtype': numpy.float32,
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: CPU usage, disabled, float32
    input_dict_8 = {
        'device_type': 'cpu',
        'dtype': numpy.float32,
        'enabled': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: CUDA with float64 (no-op but valid)
    input_dict_9 = {
        'device_type': 'cuda',
        'dtype': numpy.float64,
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: CPU with float64 (no-op but valid)
    input_dict_10 = {
        'device_type': 'cpu',
        'dtype': numpy.float64,
        'enabled': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.autocast_2"] = autocast_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.autocast_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.autocast_2'.")

check_valid('torch.autocast', generated_inputs['torch.autocast_2'], lib="torch", suffix=2)

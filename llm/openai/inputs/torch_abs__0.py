
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def abs__inputs():
    list_of_inputs = []

    input_dict = {"input": np.array([-1.5, 0.0, 2.3, -3.7], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[-5, 0, 7], [8, -10, -2]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array(-3.14159, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[[np.nan, np.inf, -np.inf], [-1.0, 0.0, 1.0]]], dtype=np.float16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([-128, -1, 0, 1, 127], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.arange(-6, 6, dtype=np.int64).reshape(2, 1, 3, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.empty((0, 5), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.empty((2, 0, 3), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    input_dict = {"input": base[:, ::2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([-0.0, 0.0, -1e308, 1e308, -1.2345e-300], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[-32768, -12345, 0, 12345, 32767]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.linspace(-3.0, 3.0, num=24).astype(np.float32).reshape(1, 2, 3, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.abs_"] = abs__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.abs_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.abs_'.")


check_valid('torch.abs_', generated_inputs['torch.abs_'], lib="torch", suffix=0)

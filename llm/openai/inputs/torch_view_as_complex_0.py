
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def view_as_complex_inputs():
    list_of_inputs = []

    input = np.ascontiguousarray(np.array([0.0, 1.0], dtype=np.float32))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.array([[1.0, -2.0]], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(3, 2))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.linspace(-5.0, 5.0, 8, dtype=np.float64).reshape(4, 2))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.random.randn(5, 2).astype(np.float32))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.zeros((6, 2), dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.full((7, 2), -3.5, dtype=np.float32))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.array([[np.nan, 0.0],
                                           [np.inf, -np.inf],
                                           [1.0, -1.0]], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.array([[1e-10, -1e-10],
                                           [1e10, -1e10],
                                           [3.14159265, -2.71828183],
                                           [-0.0, 0.0]], dtype=np.float32))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = np.ascontiguousarray(np.arange(20, dtype=np.float64).reshape(10, 2))
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.view_as_complex"] = view_as_complex_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.view_as_complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.view_as_complex'.")


check_valid('torch.view_as_complex', generated_inputs['torch.view_as_complex'], lib="torch", suffix=0)

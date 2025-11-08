
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def as_tensor_2_inputs():
    list_of_inputs = []

    data = (np.array([1, 2, 3], dtype=np.int64),)
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.array([-1, 0, 2, 5], dtype=np.int32),)
    dtype = torch.int32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.array([1.0, -2.5, 3.3], dtype=np.float32),)
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.arange(12, dtype=np.float32).reshape(3, 4),)
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.arange(24, dtype=np.float64).reshape(2, 3, 4),)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int64),)
    dtype = torch.int64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.zeros((0,), dtype=np.float32),)
    dtype = torch.float32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),)
    dtype = torch.int32
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    data = (np.linspace(-5, 5, 7, dtype=np.float64),)
    dtype = torch.float64
    list_of_inputs.append(copy.deepcopy({"data": data, "dtype": dtype}))

    return list_of_inputs

generated_inputs["torch.as_tensor_2"] = as_tensor_2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_2'.")


check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_2'], lib="torch", suffix=2)

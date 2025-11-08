
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def as_tensor_inputs():
    list_of_inputs = []

    data = np.array([1.0, 2.5, -3.0], dtype=np.float32)
    dtype = torch.float32
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1.2, 0.0, 3.14],
                     [2.71, -4.2, 5.0]], dtype=np.float64)
    dtype = torch.float64
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array(42, dtype=np.int64)
    dtype = torch.int64
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1, -2], [3, -4]],
                     [[5, -6], [7, -8]]], dtype=np.int32)
    dtype = torch.int32
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[True, False, True],
                     [False, False, True]], dtype=np.bool_)
    dtype = torch.bool
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[0, 127, 255],
                     [10, 20, 30]], dtype=np.uint8)
    dtype = torch.uint8
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1+2j, -3-4j, 0+1j], dtype=np.complex64)
    dtype = torch.complex64
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[1+0j, 2-1j],
                     [-2+3j, 4+5j]], dtype=np.complex128)
    dtype = torch.complex128
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(20, dtype=np.float32).reshape(4, 5)
    data = base[:, ::2]
    dtype = torch.float64
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    dtype = torch.float64
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([np.nan, np.inf, -np.inf, -0.0, 1.0], dtype=np.float64)
    dtype = torch.float64
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(2*1*2*3, dtype=np.float16).reshape(2, 1, 2, 3)
    dtype = torch.float16
    input_dict = {"data": data, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_tensor_3"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_3'.")


check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_3'], lib="torch", suffix=3)

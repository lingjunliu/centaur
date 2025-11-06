
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def as_tensor_inputs():
    list_of_inputs = []

    input_dict = {"data": np.int8(-128), "dtype": torch.int8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int16(1234), "dtype": torch.int16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int32(-123456), "dtype": torch.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int64(0), "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.uint8(255), "dtype": torch.uint8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int64(42), "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int32(-1), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int32(0), "dtype": torch.bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int8(-7), "dtype": torch.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int16(2), "dtype": torch.complex128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int64(1024), "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.int32(2147483647), "dtype": torch.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_tensor_4"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_tensor_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_4'.")


check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_4'], lib="torch", suffix=4)

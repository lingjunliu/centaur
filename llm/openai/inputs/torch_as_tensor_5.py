
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def as_tensor_inputs():
    list_of_inputs = []

    input_dict = {"data": np.float32(3.14), "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float64(-2.718281828), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float16(0.0), "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float32(-0.0), "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float32(1e-38), "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float64(1e-308), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float32(np.inf), "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float64(-np.inf), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float64(np.nan), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float16(np.finfo(np.float16).max), "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float32(-np.finfo(np.float32).max), "dtype": torch.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"data": np.float64(1.5e40), "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_tensor_5"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_tensor_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_5'.")


check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_5'], lib="torch", suffix=5)

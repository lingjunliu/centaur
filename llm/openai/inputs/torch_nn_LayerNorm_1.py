
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def layernorm_inputs():
    list_of_inputs = []
    
    x = np.random.randn(8).astype(np.float32)
    input_dict = {
        "normalized_shape": 8,
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": np.dtype('float32'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(4, 8).astype(np.float64)
    input_dict = {
        "normalized_shape": 8,
        "eps": 1e-6,
        "elementwise_affine": True,
        "bias": True,
        "dtype": np.dtype('float64'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict = {
        "normalized_shape": 5,
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": False,
        "dtype": np.dtype('float32'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 4, 6, 6).astype(np.float64)
    input_dict = {
        "normalized_shape": 6,
        "eps": 1e-3,
        "elementwise_affine": False,
        "bias": False,
        "dtype": np.dtype('float64'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(1, 2, 3, 4, 7).astype(np.float32)
    input_dict = {
        "normalized_shape": 7,
        "eps": 1e-4,
        "elementwise_affine": True,
        "bias": True,
        "dtype": np.dtype('float32'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(10, 1).astype(np.float64)
    input_dict = {
        "normalized_shape": 1,
        "eps": 1e-8,
        "elementwise_affine": True,
        "bias": True,
        "dtype": np.dtype('float64'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([3.14], dtype=np.float32)
    input_dict = {
        "normalized_shape": 1,
        "eps": 1e-5,
        "elementwise_affine": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.random.randn(3, 9) * 5 - 2).astype(np.float32)
    input_dict = {
        "normalized_shape": 9,
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": False,
        "dtype": np.dtype('float32'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 2, 2, 2, 10).astype(np.float64)
    input_dict = {
        "normalized_shape": 10,
        "eps": 1e-2,
        "elementwise_affine": True,
        "bias": True,
        "dtype": np.dtype('float64'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(7, 7, 7).astype(np.float32)
    input_dict = {
        "normalized_shape": 7,
        "eps": 1e-7,
        "elementwise_affine": True,
        "bias": True,
        "dtype": np.dtype('float32'),
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LayerNorm_1"] = layernorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LayerNorm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LayerNorm_1'.")


check_valid('torch.nn.LayerNorm', generated_inputs['torch.nn.LayerNorm_1'], lib="torch", suffix=1)

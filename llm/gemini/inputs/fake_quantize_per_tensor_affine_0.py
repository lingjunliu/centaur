
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def fake_quantize_per_tensor_affine_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    input_dict = {
        "input": np.random.randn(3, 4).astype(np.float32),
        "scale": 0.1,
        "zero_point": 100,
        "quant_min": 0,
        "quant_max": 255
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Integer tensor
    input_dict = {
        "input": np.random.randint(-100, 100, size=(2, 2)).astype(np.float32),
        "scale": 0.2,
        "zero_point": 0,
        "quant_min": -128,
        "quant_max": 127
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Negative scale and zero point
    input_dict = {
        "input": np.random.randn(1, 5).astype(np.float32),
        "scale": 0.05,
        "zero_point": 100,
        "quant_min": 0,
        "quant_max": 255
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Different quant_min and quant_max
    input_dict = {
        "input": np.random.randn(4, 4, 4).astype(np.float32),
        "scale": 0.15,
        "zero_point": 0,
        "quant_min": -64,
        "quant_max": 63
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: 1D tensor
    input_dict = {
        "input": np.random.randn(10).astype(np.float32),
        "scale": 0.01,
        "zero_point": 0,
        "quant_min": -128,
        "quant_max": 127
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fake_quantize_per_tensor_affine"] = fake_quantize_per_tensor_affine_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.fake_quantize_per_tensor_affine' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fake_quantize_per_tensor_affine'.")

check_valid('torch.fake_quantize_per_tensor_affine', generated_inputs['torch.fake_quantize_per_tensor_affine'], lib="torch")

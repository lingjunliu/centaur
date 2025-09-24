
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, scale and zero_point as float
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scale_tensor = np.random.rand(input_tensor.shape[1]).astype(np.float32)
    zero_point_tensor = np.random.randint(0, 255, size=(input_tensor.shape[1],)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": 0,
        "quant_max": 255
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, negative values, different quant range
    input_tensor = np.random.randn(1, 5, 5).astype(np.float32)
    scale_tensor = np.random.rand(input_tensor.shape[1]).astype(np.float32)
    zero_point_tensor = np.random.randint(-128, 127, size=(input_tensor.shape[1],)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": -128,
        "quant_max": 127
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D input tensor, smaller range
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    scale_tensor = np.random.rand(input_tensor.shape[1]).astype(np.float32)
    zero_point_tensor = np.random.randint(0, 15, size=(input_tensor.shape[1],)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": 0,
        "quant_max": 15
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 5: Larger tensor
    input_tensor = np.random.randn(4, 8, 16, 16).astype(np.float32)
    scale_tensor = np.random.rand(input_tensor.shape[1]).astype(np.float32)
    zero_point_tensor = np.random.randint(0, 255, size=(input_tensor.shape[1],)).astype(np.int64)
    input_dict = {
        "input": input_tensor,
        "scale": scale_tensor,
        "zero_point": zero_point_tensor,
        "quant_min": 0,
        "quant_max": 255
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = fake_quantize_per_channel_affine_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fake_quantize_per_channel_affine', generated_inputs)

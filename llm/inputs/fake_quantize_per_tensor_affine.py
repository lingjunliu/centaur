
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def fake_quantize_per_tensor_affine_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = 0.5
    zero_point = 0
    quant_min = -128
    quant_max = 127
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Int tensor
    input_tensor = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    scale = 0.25
    zero_point = 0
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 2D float tensor
    input_tensor = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    scale = 0.1
    zero_point = 10
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: 3D float tensor
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    scale = 0.05
    zero_point = 128
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Negative zero_point and min/max values
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scale = 0.2
    zero_point = -5
    quant_min = -100
    quant_max = 100
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Small scale
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = 0.001
    zero_point = 0
    quant_min = -128
    quant_max = 127
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: zero point close to max
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = 0.1
    zero_point = 250
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = fake_quantize_per_tensor_affine_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fake_quantize_per_tensor_affine', generated_inputs)

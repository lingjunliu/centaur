
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32 input, channel dimension is 3
    input1 = np.random.randn(1, 3, 4, 4).astype(np.float32)
    scale1 = np.random.rand(3).astype(np.float32)
    zero_point1 = np.random.randint(0, 255, size=3).astype(np.int64)

    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "quant_min": 0,
        "quant_max": 255
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Input with negative values, different shape, int8 zero_point, channel dimension is 5
    input2 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    scale2 = np.random.rand(5).astype(np.float32)
    zero_point2 = np.random.randint(-128, 127, size=5).astype(np.int64)

    input_dict2 = {
        "input": input2,
        "scale": scale2,
        "zero_point": zero_point2,
        "quant_min": -128,
        "quant_max": 127
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different quant min/max, channel dimension is 2
    input3 = np.random.randn(1, 2, 3, 3).astype(np.float32)
    scale3 = np.random.rand(2).astype(np.float32)
    zero_point3 = np.random.randint(0, 100, size=2).astype(np.int64)

    input_dict3 = {
        "input": input3,
        "scale": scale3,
        "zero_point": zero_point3,
        "quant_min": 0,
        "quant_max": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D input, channel dimension is 4. Reshape input
    input4 = np.random.randn(4).astype(np.float32).reshape(1, 4, 1, 1)
    scale4 = np.random.rand(4).astype(np.float32)
    zero_point4 = np.random.randint(-128, 127, size=4).astype(np.int64)

    input_dict4 = {
        "input": input4,
        "scale": scale4,
        "zero_point": zero_point4,
        "quant_min": -128,
        "quant_max": 127
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger input, int8 zero_point, channel dimension is 8
    input5 = np.random.randn(1, 8, 1, 1).astype(np.float32) * -1
    scale5 = np.random.rand(8).astype(np.float32)
    zero_point5 = np.random.randint(-128, 127, size=8).astype(np.int64)

    input_dict5 = {
        "input": input5,
        "scale": scale5,
        "zero_point": zero_point5,
        "quant_min": -128,
        "quant_max": 127
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

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

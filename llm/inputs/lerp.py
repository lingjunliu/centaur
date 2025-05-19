
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lerp_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensors, weight as scalar
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    end1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    weight1 = 0.5
    input_dict1 = {"input": torch.from_numpy(input1), "end": torch.from_numpy(end1), "weight": weight1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Int tensors, weight as tensor
    input2 = np.array([1, 2, 3], dtype=np.int32)
    end2 = np.array([4, 5, 6], dtype=np.int32)
    weight2 = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    input_dict2 = {"input": torch.from_numpy(input2), "end": torch.from_numpy(end2), "weight": torch.from_numpy(weight2)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 2D tensors, weight as scalar
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    end3 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    weight3 = 0.7
    input_dict3 = {"input": torch.from_numpy(input3), "end": torch.from_numpy(end3), "weight": weight3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 3D tensors, weight as tensor
    input4 = np.random.rand(2, 3, 4).astype(np.float32)
    end4 = np.random.rand(2, 3, 4).astype(np.float32)
    weight4 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict4 = {"input": torch.from_numpy(input4), "end": torch.from_numpy(end4), "weight": torch.from_numpy(weight4)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Different data types, weight as scalar
    input5 = np.array([1, 2, 3], dtype=np.int64)
    end5 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    weight5 = 0.3
    input_dict5 = {"input": torch.from_numpy(input5), "end": torch.from_numpy(end5), "weight": weight5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = lerp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lerp', generated_inputs)

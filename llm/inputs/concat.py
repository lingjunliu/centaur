
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def concat_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensors, dim=0
    tensors1 = [np.random.randn(2, 3).astype(np.float32), np.random.randn(3, 3).astype(np.float32)]
    input_dict1 = {"tensors": tensors1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensors, dim=1
    tensors2 = [np.random.randint(0, 10, size=(2, 4), dtype=np.int32), np.random.randint(0, 10, size=(2, 5), dtype=np.int32)]
    input_dict2 = {"tensors": tensors2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensors, dim=2
    tensors3 = [np.random.randn(2, 3, 4).astype(np.float64), np.random.randn(2, 3, 5).astype(np.float64)]
    input_dict3 = {"tensors": tensors3, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensors, dim=0
    tensors4 = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)]
    input_dict4 = {"tensors": tensors4, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: List of single tensor, dim = 0
    tensors5 = [np.random.randn(5, 5).astype(np.float32)]
    input_dict5 = {"tensors": tensors5, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Tensors with negative values, dim = 1
    tensors6 = [np.random.randn(2, 3).astype(np.float32), np.random.randn(2, 4).astype(np.float32)]
    input_dict6 = {"tensors": tensors6, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Different dtypes in the list
    tensors7 = [np.random.randn(2, 3).astype(np.float32), np.random.randn(2, 3).astype(np.float64)]
    input_dict7 = {"tensors": tensors7, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = concat_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('concat', generated_inputs)

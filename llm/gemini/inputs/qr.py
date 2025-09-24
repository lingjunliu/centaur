
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def qr_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    input1 = np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Rectangular matrix (m > n)
    input2 = np.random.rand(5, 3).astype(np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Rectangular matrix (m < n)
    input3 = np.random.rand(3, 5).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Batch of matrices
    input4 = np.random.rand(2, 4, 4).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Matrix with negative values
    input5 = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = qr_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('qr', generated_inputs)

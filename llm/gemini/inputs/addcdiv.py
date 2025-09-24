
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addcdiv_inputs():
    list_of_inputs = []

    input_tensor = np.random.randn(3, 4).astype(np.float32)
    tensor1 = np.random.randn(3, 4).astype(np.float32)
    tensor2 = np.random.randn(3, 4).astype(np.float32)
    value = 2.0
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.random.randn(2, 3, 4).astype(np.float64)
    tensor1 = np.random.randn(2, 3, 4).astype(np.float64)
    tensor2 = np.random.randn(2, 3, 4).astype(np.float64)
    value = -1.0
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    tensor1 = np.array([4.0, 5.0, 6.0]).astype(np.float32)
    tensor2 = np.array([7.0, 8.0, 9.0]).astype(np.float32)
    value = 1.5
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.random.randn(1, 5, 5).astype(np.float32)
    tensor1 = np.random.randn(1, 5, 5).astype(np.float32)
    tensor2 = np.random.randn(1, 5, 5).astype(np.float32)
    value = -0.5
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.random.randn(2, 2, 2, 2).astype(np.float32)
    tensor1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    tensor2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    value = 1.0
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = addcdiv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addcdiv', generated_inputs)

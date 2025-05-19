
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def baddbmm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors - Corrected shape
    input1 = np.random.randn(2, 5).astype(np.float32)
    batch1_1 = np.random.randn(2, 2, 3).astype(np.float32)
    batch2_1 = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "batch1": batch1_1,
        "batch2": batch2_1,
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different shapes, beta and alpha - Corrected shape
    input2 = np.random.randn(3, 7).astype(np.float32)
    batch1_2 = np.random.randn(3, 3, 4).astype(np.float32)
    batch2_2 = np.random.randn(3, 4, 7).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "batch1": batch1_2,
        "batch2": batch2_2,
        "beta": 0.5,
        "alpha": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))


    # Input 4: Negative values - Corrected Shape
    input4 = np.random.randn(4, 6).astype(np.float32)
    batch1_4 = np.random.randn(4, 4, 2).astype(np.float32)
    batch2_4 = np.random.randn(4, 2, 6).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "batch1": batch1_4,
        "batch2": batch2_4,
        "beta": -1.0,
        "alpha": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger batch size - Corrected Shape
    input5 = np.random.randn(6, 8).astype(np.float32)
    batch1_5 = np.random.randn(6, 6, 5).astype(np.float32)
    batch2_5 = np.random.randn(6, 5, 8).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "batch1": batch1_5,
        "batch2": batch2_5,
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different beta and alpha values - Corrected Shape
    input6 = np.random.randn(5, 7).astype(np.float32)
    batch1_6 = np.random.randn(5, 5, 3).astype(np.float32)
    batch2_6 = np.random.randn(5, 3, 7).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "batch1": batch1_6,
        "batch2": batch2_6,
        "beta": 0.7,
        "alpha": 1.3
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = baddbmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('baddbmm', generated_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addbmm_inputs():
    list_of_inputs = []

    input_np = np.random.randn(3, 10).astype(np.float32)
    batch1_np = np.random.randn(3, 10, 5).astype(np.float32)
    batch2_np = np.random.randn(3, 5, 10).astype(np.float32)
    beta = 1.0
    alpha = 2.0
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(3, 3).astype(np.float32)
    batch1_np = np.random.randn(3, 3, 2).astype(np.float32)
    batch2_np = np.random.randn(3, 2, 3).astype(np.float32)
    beta = 0.5
    alpha = 1.5
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(1, 5).astype(np.float32)
    batch1_np = np.random.randn(1, 5, 2).astype(np.float32)
    batch2_np = np.random.randn(1, 2, 5).astype(np.float32)
    beta = 0.0
    alpha = 1.0
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(4, 6).astype(np.float32)
    batch1_np = np.random.randn(4, 6, 4).astype(np.float32)
    batch2_np = np.random.randn(4, 4, 6).astype(np.float32)
    beta = -1.0
    alpha = 0.5
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.random.randn(5, 4).astype(np.float32)
    batch1_np = np.random.randn(5, 4, 5).astype(np.float32)
    batch2_np = np.random.randn(5, 5, 4).astype(np.float32)
    beta = 2.0
    alpha = -1.0
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = addbmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addbmm', list_of_inputs)

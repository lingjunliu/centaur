
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def batch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input
    input1 = np.random.randn(2, 3).astype(np.float32)
    running_mean1 = np.random.randn(3).astype(np.float32)
    running_var1 = np.random.rand(3).astype(np.float32)
    weight1 = np.random.randn(3).astype(np.float32)
    bias1 = np.random.randn(3).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "weight": weight1,
        "bias": bias1,
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D input (Batch, Channel, Length)
    input2 = np.random.randn(4, 5, 6).astype(np.float32)
    running_mean2 = np.random.randn(5).astype(np.float32)
    running_var2 = np.random.rand(5).astype(np.float32)
    weight2 = np.random.randn(5).astype(np.float32)
    bias2 = np.random.randn(5).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "running_mean": running_mean2,
        "running_var": running_var2,
        "weight": weight2,
        "bias": bias2,
        "training": True,
        "momentum": 0.2,
        "eps": 1e-8
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D input (Batch, Channel, Height, Width)
    input3 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    running_mean3 = np.random.randn(3).astype(np.float32)
    running_var3 = np.random.rand(3).astype(np.float32)
    weight3 = np.random.randn(3).astype(np.float32)
    bias3 = np.random.randn(3).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "running_mean": running_mean3,
        "running_var": running_var3,
        "weight": weight3,
        "bias": bias3,
        "training": False,
        "momentum": 0.15,
        "eps": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Input with negative values and different eps
    input4 = np.random.randn(1, 2, 3, 4).astype(np.float32) * -1
    running_mean4 = np.random.randn(2).astype(np.float32)
    running_var4 = np.random.rand(2).astype(np.float32)
    weight4 = np.random.randn(2).astype(np.float32)
    bias4 = np.random.randn(2).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "running_mean": running_mean4,
        "running_var": running_var4,
        "weight": weight4,
        "bias": bias4,
        "training": True,
        "momentum": 0.3,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Large input size
    input5 = np.random.randn(8, 16, 32, 32).astype(np.float32)
    running_mean5 = np.random.randn(16).astype(np.float32)
    running_var5 = np.random.rand(16).astype(np.float32)
    weight5 = np.random.randn(16).astype(np.float32)
    bias5 = np.random.randn(16).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "running_mean": running_mean5,
        "running_var": running_var5,
        "weight": weight5,
        "bias": bias5,
        "training": False,
        "momentum": 0.05,
        "eps": 1e-7
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = batch_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('batch_norm', generated_inputs)

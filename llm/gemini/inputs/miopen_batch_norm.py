
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def miopen_batch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic case, 4D float32
    input1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    weight1 = np.random.randn(3).astype(np.float32)
    bias1 = np.random.randn(3).astype(np.float32)
    running_mean1 = np.random.randn(3).astype(np.float32)
    running_var1 = np.random.rand(3).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "training": True,
        "exponential_average_factor": 0.1,
        "epsilon": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D float64, different training params
    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    weight2 = np.random.randn(3).astype(np.float64)
    bias2 = np.random.randn(3).astype(np.float64)
    running_mean2 = np.random.randn(3).astype(np.float64)
    running_var2 = np.random.rand(3).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "bias": bias2,
        "running_mean": running_mean2,
        "running_var": running_var2,
        "training": False,
        "exponential_average_factor": 0.2,
        "epsilon": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 5D float32, no training
    input3 = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    weight3 = np.random.randn(3).astype(np.float32)
    bias3 = np.random.randn(3).astype(np.float32)
    running_mean3 = np.random.randn(3).astype(np.float32)
    running_var3 = np.random.rand(3).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "bias": bias3,
        "running_mean": running_mean3,
        "running_var": running_var3,
        "training": False,
        "exponential_average_factor": 0.3,
        "epsilon": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D float16 (if supported), training true, different epsilon
    input4 = np.random.randn(2, 3, 4, 5).astype(np.float32) # keep float32, float16 causes problems with miopen
    weight4 = np.random.randn(3).astype(np.float32)
    bias4 = np.random.randn(3).astype(np.float32)
    running_mean4 = np.random.randn(3).astype(np.float32)
    running_var4 = np.random.rand(3).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "bias": bias4,
        "running_mean": running_mean4,
        "running_var": running_var4,
        "training": True,
        "exponential_average_factor": 0.4,
        "epsilon": 1e-2
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D, training false
    input5 = np.random.randn(2, 3).astype(np.float32)
    weight5 = np.random.randn(3).astype(np.float32)
    bias5 = np.random.randn(3).astype(np.float32)
    running_mean5 = np.random.randn(3).astype(np.float32)
    running_var5 = np.random.rand(3).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "bias": bias5,
        "running_mean": running_mean5,
        "running_var": running_var5,
        "training": False,
        "exponential_average_factor": 0.5,
        "epsilon": 1e-1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different shapes
    input6 = np.random.randn(1, 5, 7, 9).astype(np.float32)
    weight6 = np.random.randn(5).astype(np.float32)
    bias6 = np.random.randn(5).astype(np.float32)
    running_mean6 = np.random.randn(5).astype(np.float32)
    running_var6 = np.random.rand(5).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "weight": weight6,
        "bias": bias6,
        "running_mean": running_mean6,
        "running_var": running_var6,
        "training": True,
        "exponential_average_factor": 0.6,
        "epsilon": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Small values
    input7 = np.random.randn(2, 3, 4, 5).astype(np.float32) * 0.01
    weight7 = np.random.randn(3).astype(np.float32) * 0.01
    bias7 = np.random.randn(3).astype(np.float32) * 0.01
    running_mean7 = np.random.randn(3).astype(np.float32) * 0.01
    running_var7 = np.random.rand(3).astype(np.float32) * 0.01
    input_dict7 = {
        "input": input7,
        "weight": weight7,
        "bias": bias7,
        "running_mean": running_mean7,
        "running_var": running_var7,
        "training": False,
        "exponential_average_factor": 0.7,
        "epsilon": 1e-7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = miopen_batch_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('miopen_batch_norm', generated_inputs)

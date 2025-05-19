
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def PoissonNLLLoss_inputs():
    list_of_inputs = []

    # Input 1: Basic example with log_input=True, full=False
    input1 = np.random.randn(3, 5).astype(np.float32)
    target1 = np.random.randint(0, 10, size=(3, 5)).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "log_input": True,
        "full": False,
        "eps": 1e-8,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: log_input=False, full=True, different reduction
    input2 = np.random.rand(2, 4, 6).astype(np.float64) * 10  # Positive values for exp
    target2 = np.random.randint(0, 5, size=(2, 4, 6)).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "log_input": False,
        "full": True,
        "eps": 1e-6,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different shape, log_input=True, full=False
    input3 = np.random.randn(1, 7, 7).astype(np.float32)
    target3 = np.random.randint(0, 8, size=(1, 7, 7)).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "target": target3,
        "log_input": True,
        "full": False,
        "eps": 1e-10,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar input and target, log_input=False, full=True
    input4 = np.array(5.0).astype(np.float64)
    target4 = np.array(2).astype(np.float64)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "log_input": False,
        "full": True,
        "eps": 1e-5,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 1D tensor
    input5 = np.random.randn(10).astype(np.float32)
    target5 = np.random.randint(0, 10, size=(10)).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "target": target5,
        "log_input": True,
        "full": False,
        "eps": 1e-8,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = PoissonNLLLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PoissonNLLLoss', generated_inputs)

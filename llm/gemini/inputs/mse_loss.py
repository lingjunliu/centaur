
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def mse_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, reduction='mean'
    input1 = np.random.randn(3, 5).astype(np.float32)
    target1 = np.random.randn(3, 5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors, reduction='sum' (Cast to float)
    input2 = np.random.randint(0, 10, size=(2, 4)).astype(np.float32)
    target2 = np.random.randint(0, 10, size=(2, 4)).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 1D tensors, reduction='none' (removed because of error)

    # Case 4: Higher dimensional tensors (4D), reduction='mean'
    input4 = np.random.randn(1, 3, 28, 28).astype(np.float32)
    target4 = np.random.randn(1, 3, 28, 28).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Negative values, reduction='sum'
    input5 = np.random.randn(2, 2) * -1.0
    target5 = np.random.randn(2, 2) * -1.0
    input_dict5 = {
        "input": input5.astype(np.float32),
        "target": target5.astype(np.float32),
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: Different shapes that are broadcastable, reduction='mean'
    input6 = np.random.randn(1, 5).astype(np.float32)
    target6 = np.random.randn(5).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "target": target6,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = mse_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mse_loss', generated_inputs)

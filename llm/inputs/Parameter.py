
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def parameter_inputs():
    generated_inputs = []

    # Input 1: Float tensor, requires_grad=True
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor, requires_grad=False
    data = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int tensor, requires_grad=True.  This is likely invalid.
    # data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    # input_dict = {"data": data, "requires_grad": True}
    # generated_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative float tensor, requires_grad=True
    data = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional float tensor, requires_grad=False
    data = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"data": data, "requires_grad": False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero tensor, requires_grad=True
    data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element tensor, requires_grad=True
    data = np.array(3.14, dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = parameter_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Parameter', generated_inputs)

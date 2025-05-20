
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D numpy array of integers
    obj = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"obj": obj, "dtype": None, "copy": None, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D numpy array of floats, specify dtype
    obj = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict = {"obj": obj, "dtype": torch.float64, "copy": True, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Numpy scalar
    obj = np.float32(0.5)
    input_dict = {"obj": obj, "dtype": None, "copy": None, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D numpy array of negative integers
    obj = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"obj": obj, "dtype": None, "copy": None, "requires_grad": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = asarray_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('asarray', generated_inputs)

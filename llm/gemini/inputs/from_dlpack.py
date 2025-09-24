
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def from_dlpack_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    dlpack_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"dlpack": dlpack_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor, 2D
    dlpack_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"dlpack": dlpack_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = from_dlpack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('from_dlpack', generated_inputs)

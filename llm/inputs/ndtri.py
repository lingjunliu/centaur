
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ndtri_inputs():
    generated_inputs = []

    # Input 1: Basic float array
    x1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    generated_inputs.append({"x": x1})

    # Input 2: Different float values including near 0 and 1
    x2 = np.array([0.01, 0.99, 0.25, 0.75], dtype=np.float64)
    generated_inputs.append({"x": x2})

    # Input 3: Multi-dimensional float array
    x3 = np.array([[0.2, 0.3], [0.6, 0.8]], dtype=np.float32)
    generated_inputs.append({"x": x3})

    # Input 4: Array with a value of 0.5
    x4 = np.array([0.5], dtype=np.float64)
    generated_inputs.append({"x": x4})

    # Input 5: Higher dimensional array
    x5 = np.random.rand(2, 3, 4).astype(np.float32)
    generated_inputs.append({"x": x5})


    return generated_inputs

generated_inputs = ndtri_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ndtri', generated_inputs)

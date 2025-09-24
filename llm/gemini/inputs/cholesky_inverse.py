
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cholesky_inverse_inputs():
    generated_inputs = []

    # Input 1: Basic float32, lower=False, positive definite
    L = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": False})

    # Input 2: float64, upper=True, positive definite
    L = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]], dtype=np.float64)
    generated_inputs.append({"L": L, "upper": True})

    # Input 3: Batched input (2 batches), lower=False, positive definite
    L = np.array([[[2.0, 0.0, 0.0],
                   [0.0, 3.0, 0.0],
                   [0.0, 0.0, 4.0]],
                  [[1.0, 0.0, 0.0],
                   [0.0, 4.0, 0.0],
                   [0.0, 0.0, 6.0]]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": False})
    
    # Input 4: Different size matrix, float32, upper=True, positive definite
    L = np.array([[4.0, 0.0],
                  [0.0, 2.0]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": True})

    # Input 5: Larger matrix, lower=False, positive definite
    L = np.array([[5.0, 0.0, 0.0, 0.0],
                  [0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 7.0, 0.0],
                  [0.0, 0.0, 0.0, 8.0]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": False})
    
    return generated_inputs

generated_inputs = cholesky_inverse_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky_inverse', generated_inputs)

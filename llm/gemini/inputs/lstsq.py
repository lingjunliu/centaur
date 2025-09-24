
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lstsq_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    A = np.random.randn(5, 3).astype(np.float32)
    B = np.random.randn(5, 2).astype(np.float32)
    rcond = 1e-15
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, double tensors
    A = np.random.randn(10, 5).astype(np.float64)
    B = np.random.randn(10, 1).astype(np.float64)
    rcond = 1e-8
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Overdetermined system
    A = np.random.randn(10, 3).astype(np.float64)
    B = np.random.randn(10, 5).astype(np.float64)
    rcond = 1e-12
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Underdetermined system
    A = np.random.randn(3, 10).astype(np.float32)
    B = np.random.randn(3, 2).astype(np.float32)
    rcond = 1e-6
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A is square matrix, B is vector
    A = np.random.randn(5, 5).astype(np.float32)
    B = np.random.randn(5).astype(np.float32)
    rcond = 1e-14
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lstsq_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lstsq', generated_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cholesky_solve_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    A = np.array([[4.0, 12.0, -16.0],
                  [12.0, 37.0, -43.0],
                  [-16.0, -43.0, 98.0]], dtype=np.float32)
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    input_dict = {
        "input": b,
        "input2": A
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = cholesky_solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky_solve', generated_inputs)

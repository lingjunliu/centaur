
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def triangular_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    A = np.random.rand(3, 3).astype(np.float32)
    b = np.random.rand(3, 1).astype(np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Upper triangular, transposed, unitriangular
    A = np.triu(np.random.rand(5, 5)).astype(np.float64)
    b = np.random.rand(5, 3).astype(np.float64)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": True,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensors
    A = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    b = (np.random.rand(2, 1) + 1j * np.random.rand(2, 1)).astype(np.complex64)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched input, negative values
    A = np.tril(np.random.randn(2, 3, 3)).astype(np.float32)
    b = np.random.randn(2, 3, 2).astype(np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimension b - Correcting dimension issue - Second attempt
    A = np.random.rand(4, 4).astype(np.float32)
    b = np.random.rand(4, 2).astype(np.float32) # Corrected dimensions to be compatible with A
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = triangular_solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('triangular_solve', generated_inputs)

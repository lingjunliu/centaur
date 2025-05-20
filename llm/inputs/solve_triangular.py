
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, upper triangular
    A = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    b = np.array([[8.0], [9.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower triangular, float tensor
    A = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    b = np.array([[8.0], [9.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unitriangular, transposed
    A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[8.0], [9.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": True,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Multi-dimensional input
    A = np.array([[[2.0, 1.0], [0.0, 3.0]], [[1.0, 2.0], [0.0, 4.0]]], dtype=np.float32)
    b = np.array([[[8.0], [9.0]], [[5.0], [6.0]]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex tensor
    A = np.array([[2.0 + 1j, 1.0], [0.0, 3.0 - 2j]], dtype=np.complex64)
    b = np.array([[8.0 + 2j], [9.0 - 1j]], dtype=np.complex64)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another multi-dimensional example, lower triangular
    A = np.array([[[1, 0], [2, 3]], [[4, 0], [5, 6]]], dtype=np.float32)
    b = np.array([[[7], [8]], [[9], [10]]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = solve_triangular_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('solve_triangular', generated_inputs)

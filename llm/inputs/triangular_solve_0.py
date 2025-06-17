
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def triangular_solve_inputs():
    list_of_inputs = []

    # Case 1: Basic float input
    A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 1.0, 4.0]], dtype=np.float32)
    b = np.array([[8.0], [11.0], [20.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "input": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Upper triangular, transposed, unitriangular
    A = np.array([[1.0, 1.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    b = np.array([[8.0], [11.0], [20.0]], dtype=np.float64)
    input_dict = {
        "A": A,
        "input": b,
        "upper": True,
        "transpose": True,
        "unitriangular": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Multiple vectors
    A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 1.0, 4.0]], dtype=np.float32)
    b = np.array([[8.0, 11.0, 20.0], [4.0, 5.0, 6.0]], dtype=np.float32).T
    input_dict = {
        "A": A,
        "input": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Double precision
    A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    b = np.array([[8.0], [11.0], [20.0]], dtype=np.float64)
    input_dict = {
        "A": A,
        "input": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Complex input
    A = np.array([[2.0j, 0.0, 0.0], [1.0, 3.0j, 0.0], [2.0, 1.0, 4.0j]], dtype=np.complex64)
    b = np.array([[8.0j], [11.0], [20.0]], dtype=np.complex64)
    input_dict = {
        "A": A,
        "input": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 2D input and 2D A
    A = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    input_dict = {
        "A": A,
        "input": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.triangular_solve"] = triangular_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.triangular_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.triangular_solve'.")

check_valid('torch.triangular_solve', generated_inputs['torch.triangular_solve'], lib="torch")

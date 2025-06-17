
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic float upper triangular matrix
    a = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    b = np.array([8.0, 15.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": True,
        "unitriangular": False,
        "left": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer lower triangular matrix
    a = np.array([[2, 0], [1, 3]], dtype=np.int32)
    b = np.array([8, 15], dtype=np.int32)
    input_dict = {
        "a": a.astype(np.float32),
        "b": b.astype(np.float32),
        "upper": False,
        "unitriangular": False,
        "left": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unitriangular lower matrix
    a = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    b = np.array([8.0, 15.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": False,
        "unitriangular": True,
        "left": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4: Different sized matrix
    a = np.array([[2.0, 1.0, 3.0], [0.0, 3.0, 4.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    b = np.array([8.0, 15.0, 25.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": True,
        "unitriangular": False,
        "left": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    a = np.array([[2.0j, 1.0], [0.0, 3.0j]], dtype=np.complex64)
    b = np.array([8.0, 15.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "upper": True,
        "unitriangular": False,
        "left": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.solve_triangular"] = solve_triangular_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.solve_triangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve_triangular'.")

check_valid('torch.linalg.solve_triangular', generated_inputs['torch.linalg.solve_triangular'], lib="torch")

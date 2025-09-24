
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic upper triangular matrix
    a = np.array([[1.0, 2.0, 3.0],
                  [0.0, 4.0, 5.0],
                  [0.0, 0.0, 6.0]], dtype=np.float32)
    b = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": np.bool_(True),
        "unitriangular": np.bool_(False),
        "left": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower triangular matrix
    a = np.array([[1.0, 0.0, 0.0],
                  [2.0, 3.0, 0.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": np.bool_(False),
        "unitriangular": np.bool_(False),
        "left": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unitriangular matrix
    a = np.array([[1.0, 2.0, 3.0],
                  [0.0, 1.0, 5.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    b = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": np.bool_(True),
        "unitriangular": np.bool_(True),
        "left": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Right-hand side
    a = np.array([[1.0, 2.0, 3.0],
                  [0.0, 4.0, 5.0],
                  [0.0, 0.0, 6.0]], dtype=np.float32)
    b = np.array([[7.0, 8.0],
                  [9.0, 10.0],
                  [11.0, 12.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "upper": np.bool_(True),
        "unitriangular": np.bool_(False),
        "left": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: b is 1D
    a = np.array([[1.0, 2.0, 3.0],
                  [0.0, 4.0, 5.0],
                  [0.0, 0.0, 6.0]], dtype=np.float32)
    b = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    input_dict = {
        "a": a,
        "b": b,
        "upper": np.bool_(True),
        "unitriangular": np.bool_(False),
        "left": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.solve_triangular"] = solve_triangular_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.solve_triangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve_triangular'.")

check_valid('torch.linalg.solve_triangular', generated_inputs['torch.linalg.solve_triangular'], lib="torch", suffix=0)

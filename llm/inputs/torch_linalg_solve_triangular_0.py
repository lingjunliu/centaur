
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic upper triangular
    a = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float32)
    b = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic lower triangular
    a = np.array([[1, 0, 0], [2, 4, 0], [3, 5, 6]], dtype=np.float32)
    b = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": False, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unitriangular
    a = np.array([[1, 2, 3], [0, 1, 5], [0, 0, 1]], dtype=np.float32)
    b = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": True, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Right-hand side matrix
    a = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float32)
    b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different dtype
    a = np.array([[1, 2], [0, 4]], dtype=np.float64)
    b = np.array([1, 2], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Left=False
    a = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float32)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex dtype
    a = np.array([[1+1j, 2], [0, 4+2j]], dtype=np.complex64)
    b = np.array([1+1j, 2+2j], dtype=np.complex64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: a and b as tensors (already numpy arrays)
    a = np.array([[1, 2], [0, 4]], dtype=np.float32)
    b = np.array([1, 2], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: b is matrix
    a = np.array([[1, 2], [0, 4]], dtype=np.float32)
    b = np.array([[1, 2], [3,4]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: a and b are single element arrays
    a = np.array([[5]], dtype=np.float32)
    b = np.array([10], dtype=np.float32)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Non-contiguous array
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)[:, :2]
    b = np.array([1, 2, 3], dtype=np.float32)[:2]
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
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

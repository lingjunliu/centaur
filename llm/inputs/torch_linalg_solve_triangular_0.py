
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic example with upper triangular matrix
    a = np.array([[2, 1], [0, 3]], dtype=np.float64)
    b = np.array([8, 5], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower triangular matrix
    a = np.array([[2, 0], [1, 3]], dtype=np.float64)
    b = np.array([8, 5], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": False, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unitriangular matrix
    a = np.array([[1, 1], [0, 1]], dtype=np.float64)
    b = np.array([8, 5], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": True, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple right-hand sides
    a = np.array([[2, 1], [0, 3]], dtype=np.float64)
    b = np.array([[8, 5], [10, 7]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different dtype
    a = np.array([[2, 1], [0, 3]], dtype=np.complex128)
    b = np.array([8, 5], dtype=np.complex128)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: a is a batch of matrices
    a = np.array([[[2, 1], [0, 3]], [[1, 2], [0, 4]]], dtype=np.float64)
    b = np.array([[8, 5], [10, 7]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: right is false
    a = np.array([[2, 1], [0, 3]], dtype=np.float64)
    b = np.array([[8, 5], [10, 7]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "upper": True, "unitriangular": False, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative values
    a = np.array([[2, -1], [0, 3]], dtype=np.float64)
    b = np.array([8, 5], dtype=np.float64)
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

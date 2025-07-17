
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_svd_inputs():
    list_of_inputs = []

    # Input 1: Basic example with a small matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    full_matrices = True
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: full_matrices = False
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    full_matrices = False
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A rectangular matrix (m < n)
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    full_matrices = True
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A batch of matrices
    A = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    full_matrices = True
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Different shaped batch
    A = np.random.rand(2,3,4)
    full_matrices = False
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A with driver specified (if CUDA available)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    full_matrices = True
    driver = 'gesvdj' if torch.cuda.is_available() else None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: A larger matrix
    A = np.random.rand(10, 5)
    full_matrices = False
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: A complex matrix
    A = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 - 2j]], dtype=np.complex128)
    full_matrices = True
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Another batch example
    A = np.random.rand(4, 2, 2)
    full_matrices = False
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Double type
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    full_matrices = True
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Negative Values
    A = np.array([[-1.0, 2.0], [3.0, -4.0]])
    full_matrices = True
    driver = None
    out = None
    input_dict = {"A": A, "full_matrices": full_matrices, "driver": driver, "out": out}
    input_dict["out"] = None
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.svd"] = linalg_svd_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svd'.")

check_valid('torch.linalg.svd', generated_inputs['torch.linalg.svd'], lib="torch", suffix=0)

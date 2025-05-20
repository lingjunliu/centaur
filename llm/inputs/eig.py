
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def eig_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 float matrix, eigenvectors=True
    input1 = np.random.rand(2, 2).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "eigenvectors": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Simple 3x3 float matrix, eigenvectors=False
    input2 = np.random.rand(3, 3).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "eigenvectors": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex matrix, eigenvectors=True
    input3 = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    input_dict3 = {
        "input": input3,
        "eigenvectors": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger matrix, eigenvectors=False
    input4 = np.random.rand(5, 5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "eigenvectors": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Matrix with negative values, eigenvectors=True
    input5 = (np.random.rand(3, 3) - 0.5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "eigenvectors": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex matrix, eigenvectors=False
    input6 = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex128)
    input_dict6 = {
        "input": input6,
        "eigenvectors": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Non-square matrix (should raise an error), eigenvectors=True
    input7 = np.random.rand(2, 3).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "eigenvectors": True
    }
    # This one is skipped because the error case is to be handeled in testing
    #list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs = eig_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eig', generated_inputs)

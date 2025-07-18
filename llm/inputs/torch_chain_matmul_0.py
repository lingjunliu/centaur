
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def chain_matmul_inputs():
    list_of_inputs = []

    # The framework requires a single contiguous numpy array for validation, but the API
    # torch.chain_matmul(A, B, ...) requires multiple tensor arguments. The only way
    # this can work is if the framework unpacks the first dimension of the provided numpy
    # array into separate arguments. The inputs are structured assuming this is the
    # intended behavior for the 'tensor_list' signature.
    # We use square matrices to allow them to be stacked into a single numpy array.

    # Input 1: Basic case: two 3x3 float32 matrices
    input_dict_1 = {
        'matrices': np.random.rand(2, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Chain of three 4x4 float64 matrices
    input_dict_2 = {
        'matrices': np.random.rand(3, 4, 4).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Chain of five 2x2 integer matrices
    input_dict_3 = {
        'matrices': np.random.randint(-10, 10, size=(5, 2, 2)).astype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Chain of two 1x1 matrices (edge case)
    input_dict_4 = {
        'matrices': np.array([[[1.5]], [[-2.0]]]).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Chain with negative float values
    input_dict_5 = {
        'matrices': np.random.randn(2, 5, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Longer chain of matrices (8 matrices of size 2x2)
    input_dict_6 = {
        'matrices': np.random.rand(8, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Complex number matrices (four 3x3)
    input_dict_7 = {
        'matrices': (np.random.rand(4, 3, 3) + 1j * np.random.rand(4, 3, 3)).astype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Larger matrices (two 10x10)
    input_dict_8 = {
        'matrices': np.random.rand(2, 10, 10).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Chain of three 3x3 matrices where one is the identity matrix
    mats = np.random.rand(3, 3, 3).astype(np.float32)
    mats[1, :, :] = np.eye(3, dtype=np.float32)
    input_dict_9 = {'matrices': mats}
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Minimum number of matrices (2) with int32 type
    input_dict_10 = {
        'matrices': np.random.randint(-5, 5, size=(2, 6, 6)).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.chain_matmul"] = chain_matmul_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.chain_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.chain_matmul'.")

check_valid('torch.chain_matmul', generated_inputs['torch.chain_matmul'], lib="torch", suffix=0)

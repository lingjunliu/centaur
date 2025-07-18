
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
import scipy.sparse

def smm_inputs():
    list_of_inputs = []

    def to_coo(arr):
        """Helper to create a scipy.sparse.coo_matrix from a dense numpy array."""
        return scipy.sparse.coo_matrix(arr)

    # Input 1: Basic case (float32)
    input_dense_1 = np.zeros((4, 5), dtype=np.float32)
    input_dense_1[0, 1] = 1.0
    input_dense_1[2, 3] = 2.0
    input_dense_1[3, 0] = -1.0
    mat_1 = np.random.rand(5, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_1), 'mat': mat_1}))

    # Input 2: Basic case (float64)
    input_dense_2 = np.zeros((6, 2), dtype=np.float64)
    input_dense_2[1, 1] = 5.5
    input_dense_2[4, 0] = 3.14
    mat_2 = np.random.rand(2, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_2), 'mat': mat_2}))

    # Input 3: Sparse matrix with only negative values
    input_dense_3 = np.zeros((3, 3), dtype=np.float32)
    input_dense_3[0, 0] = -1.0
    input_dense_3[1, 2] = -5.0
    input_dense_3[2, 1] = -2.5
    mat_3 = np.abs(np.random.rand(3, 3).astype(np.float32))
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_3), 'mat': mat_3}))

    # Input 4: Dense matrix with negative values
    input_dense_4 = np.zeros((5, 7), dtype=np.float32)
    input_dense_4[0, 6] = 1.0
    input_dense_4[2, 2] = 2.0
    mat_4 = np.random.rand(7, 2).astype(np.float32) - 0.8
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_4), 'mat': mat_4}))

    # Input 5: Both matrices with mixed positive/negative values
    input_dense_5 = np.zeros((8, 3), dtype=np.float64)
    input_dense_5[1, 1] = -1.0
    input_dense_5[7, 2] = 2.0
    input_dense_5[3, 0] = 3.0
    mat_5 = (np.random.rand(3, 6) - 0.5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_5), 'mat': mat_5}))

    # Input 6: Very sparse matrix (one non-zero element)
    input_dense_6 = np.zeros((10, 10), dtype=np.float32)
    input_dense_6[3, 7] = 42.0
    mat_6 = np.random.rand(10, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_6), 'mat': mat_6}))

    # Input 7: Identity sparse matrix
    input_7 = scipy.sparse.identity(7, dtype=np.float64, format='coo')
    mat_7 = np.random.rand(7, 7).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({'input': input_7, 'mat': mat_7}))

    # Input 8: Zero sparse matrix
    input_8 = scipy.sparse.coo_matrix((4, 6), dtype=np.float32)
    mat_8 = np.random.rand(6, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'input': input_8, 'mat': mat_8}))

    # Input 9: Larger dimensions
    input_dense_9 = np.zeros((50, 20), dtype=np.float32)
    input_dense_9[10, 5] = 1.0
    input_dense_9[25, 15] = -2.0
    input_dense_9[49, 19] = 3.0
    mat_9 = np.random.rand(20, 30).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_9), 'mat': mat_9}))

    # Input 10: mat is a column vector (N x 1)
    input_dense_10 = np.zeros((3, 8), dtype=np.float32)
    input_dense_10[0, 1] = 1.0
    input_dense_10[1, 4] = 2.0
    input_dense_10[2, 7] = 3.0
    mat_10 = np.random.rand(8, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({'input': to_coo(input_dense_10), 'mat': mat_10}))
    
    return list_of_inputs

generated_inputs["torch.smm"] = smm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.smm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.smm'.")

check_valid('torch.smm', generated_inputs['torch.smm'], lib="torch", suffix=0)

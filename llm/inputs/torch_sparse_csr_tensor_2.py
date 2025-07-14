
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_csr_tensor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x3 sparse tensor
    crow_indices = np.array([0, 2, 3], dtype=np.int64)
    col_indices = np.array([0, 2, 1], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    size = [2, 3]
    dtype = np.float32
    requires_grad = False

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty sparse tensor
    crow_indices = np.array([0, 0, 0], dtype=np.int64)
    col_indices = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float64)
    size = [2, 3]
    dtype = np.float64
    requires_grad = True

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float64),
        "size": size,
        "dtype": torch.float64,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dtype (int32)
    crow_indices = np.array([0, 1, 3], dtype=np.int64)
    col_indices = np.array([1, 0, 2], dtype=np.int64)
    values = np.array([4, 5, 6], dtype=np.int32)
    size = [3, 4]
    dtype = np.int32
    requires_grad = False
    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.int32),
        "size": size,
        "dtype": torch.int32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Larger size
    crow_indices = np.array([0, 2, 4, 5], dtype=np.int64)
    col_indices = np.array([0, 1, 2, 3, 1], dtype=np.int64)
    values = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    size = [4, 5]
    dtype = np.float32
    requires_grad = True

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    crow_indices = np.array([0, 1, 3], dtype=np.int64)
    col_indices = np.array([0, 0, 2], dtype=np.int64)
    values = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    size = [3, 3]
    dtype = np.float32
    requires_grad = False

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All values on the same row
    crow_indices = np.array([0, 3, 3], dtype=np.int64)
    col_indices = np.array([0, 1, 2], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    size = [2, 4]
    dtype = np.float64
    requires_grad = True

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float64),
        "size": size,
        "dtype": torch.float64,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: size with zero dimension
    crow_indices = np.array([0, 0], dtype=np.int64)
    col_indices = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float32)
    size = [2, 0]
    dtype = np.float32
    requires_grad = False

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: single element tensor
    crow_indices = np.array([0, 1], dtype=np.int64)
    col_indices = np.array([0], dtype=np.int64)
    values = np.array([5.0], dtype=np.float32)
    size = [1, 1]
    dtype = np.float32
    requires_grad = True

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large sparse tensor
    crow_indices = np.array([0, 5, 10], dtype=np.int64)
    col_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    size = [2, 10]
    dtype = np.float32
    requires_grad = False

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: sparse tensor with only one row having nonzero elements
    crow_indices = np.array([0, 0, 3], dtype=np.int64)
    col_indices = np.array([0, 1, 2], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.float32)
    size = [3, 4]
    dtype = np.float32
    requires_grad = False

    input_dict = {
        "crow_indices": torch.tensor(crow_indices, dtype=torch.int64),
        "col_indices": torch.tensor(col_indices, dtype=torch.int64),
        "values": torch.tensor(values, dtype=torch.float32),
        "size": size,
        "dtype": torch.float32,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sparse_csr_tensor_2"] = sparse_csr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_csr_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_csr_tensor_2'.")

check_valid('torch.sparse_csr_tensor', generated_inputs['torch.sparse_csr_tensor_2'], lib="torch", suffix=2)

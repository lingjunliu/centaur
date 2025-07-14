
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_csr_tensor_inputs():
    list_of_inputs = []

    # Input 1
    crow_indices = np.array([0, 2, 4])
    col_indices = np.array([0, 2, 1, 2])
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    size = (2, 3)
    dtype = np.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    crow_indices = np.array([0, 1, 2, 2])
    col_indices = np.array([2, 0])
    values = np.array([5, 6], dtype=np.float32)
    size = (4, 4)
    dtype = np.float32
    requires_grad = True
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    crow_indices = np.array([0, 0, 0])
    col_indices = np.array([])
    values = np.array([], dtype=np.float32)
    size = (3, 5)
    dtype = np.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    crow_indices = np.array([0, 1, 3])
    col_indices = np.array([0, 1, 2])
    values = np.array([-1, -2, -3], dtype=np.float32)
    size = (2, 3)
    dtype = np.float32
    requires_grad = True
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    crow_indices = np.array([0, 2, 3])
    col_indices = np.array([1, 3, 0])
    values = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    size = (3, 4)
    dtype = np.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    crow_indices = np.array([0, 1, 2, 3])
    col_indices = np.array([0, 1, 2])
    values = np.array([1, 2, 3], dtype=np.float32)
    size = (4, 3)
    dtype = np.float32
    requires_grad = True
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty values
    crow_indices = np.array([0, 0, 0])
    col_indices = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float32)
    size = (3, 3)
    dtype = np.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    crow_indices = np.array([0, 1])
    col_indices = np.array([0])
    values = np.array([1], dtype=np.float32)
    size = (2, 1)
    dtype = np.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    crow_indices = np.array([0, 0])
    col_indices = np.array([])
    values = np.array([], dtype=np.float32)
    size = (2, 2)
    dtype = np.float32
    requires_grad = True
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    crow_indices = np.array([0, 3, 5])
    col_indices = np.array([0, 1, 2, 0, 1])
    values = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    size = (2, 3)
    dtype = np.float32
    requires_grad = False
    input_dict = {
        "crow_indices": crow_indices,
        "col_indices": col_indices,
        "values": values,
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    final_list = []
    for input_dict in list_of_inputs:
        final_dict = {}
        final_dict['crow_indices'] = torch.tensor(input_dict['crow_indices'], dtype=torch.int64)
        final_dict['col_indices'] = torch.tensor(input_dict['col_indices'], dtype=torch.int64)
        final_dict['values'] = torch.tensor(input_dict['values'])
        final_dict['size'] = input_dict['size']
        final_dict['dtype'] = torch.float32
        final_dict['requires_grad'] = input_dict['requires_grad']

        final_list.append(final_dict)

    return final_list

generated_inputs["torch.sparse_csr_tensor_1"] = sparse_csr_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sparse_csr_tensor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_csr_tensor_1'.")

check_valid('torch.sparse_csr_tensor', generated_inputs['torch.sparse_csr_tensor_1'], lib="torch", suffix=1)

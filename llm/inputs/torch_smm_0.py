
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import scipy.sparse as sp

def smm_inputs():
    list_of_inputs = []

    # Input 1
    sparse_matrix = sp.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 2
    sparse_matrix = sp.csr_matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 3
    sparse_matrix = sp.csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 4
    sparse_matrix = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 5
    sparse_matrix = sp.csr_matrix([[1, 0, 0], [0, 0, 0], [0, 0, 5]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 6
    sparse_matrix = sp.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 7
    sparse_matrix = sp.csr_matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 8
    sparse_matrix = sp.csr_matrix([[2, 0, 0], [0, 4, 0], [0, 0, 6]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 9
    sparse_matrix = sp.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    # Input 10
    sparse_matrix = sp.csr_matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)
    
    # Input 11: Larger sparse matrix
    sparse_matrix = sp.random(100, 50, density=0.1, format="csr")
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), sparse_matrix.shape)
    dense_matrix = np.random.rand(50, 20).astype(np.float32)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)
    
    # Input 12: Float64
    sparse_matrix = sp.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.DoubleTensor(values), sparse_matrix.shape)
    dense_matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    input_dict = {"input": sparse_tensor, "mat": dense_matrix}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["torch.smm"] = smm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.smm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.smm'.")

check_valid('torch.smm', generated_inputs['torch.smm'], lib="torch", suffix=0)

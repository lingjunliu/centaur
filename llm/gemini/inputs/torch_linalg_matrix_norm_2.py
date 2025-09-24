
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def linalg_matrix_norm_inputs():
    list_of_inputs = []

    # Input 1: ord='fro', basic 2D, float32
    input_1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    out_1 = np.zeros((), dtype=np.float32)
    input_dict_1 = {
        'input': input_1,
        'ord': 'fro',
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_1,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: ord='nuc', basic 2D, float64, keepdim=True
    input_2 = np.random.randn(3, 4).astype(np.float64)
    out_2 = np.zeros((1, 1), dtype=np.float64)
    input_dict_2 = {
        'input': input_2,
        'ord': 'nuc',
        'dim': (-2, -1),
        'keepdim': True,
        'out': out_2,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: ord='fro', 3D tensor, keepdim=False
    input_3 = np.random.rand(5, 3, 2).astype(np.float32)
    out_3 = np.zeros((5,), dtype=np.float32)
    input_dict_3 = {
        'input': input_3,
        'ord': 'fro',
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_3,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: ord='nuc', 3D tensor, keepdim=True, non-default dim
    input_4 = np.random.rand(5, 3, 2).astype(np.float64)
    out_4 = np.zeros((2, 1, 1), dtype=np.float64)
    input_dict_4 = {
        'input': input_4,
        'ord': 'nuc',
        'dim': (0, 1),
        'keepdim': True,
        'out': out_4,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: ord='fro', 4D tensor, non-default dim
    input_5 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    out_5 = np.zeros((3, 5), dtype=np.float32)
    input_dict_5 = {
        'input': input_5,
        'ord': 'fro',
        'dim': (0, 2),
        'keepdim': False,
        'out': out_5,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: ord='nuc', 4D tensor, non-default dim, keepdim=True
    input_6 = np.random.rand(2, 3, 4, 5).astype(np.float64)
    out_6 = np.zeros((2, 1, 1, 5), dtype=np.float64)
    input_dict_6 = {
        'input': input_6,
        'ord': 'nuc',
        'dim': (1, 2),
        'keepdim': True,
        'out': out_6,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: ord='fro', complex64 input, keepdim=False
    input_7 = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex64)
    out_7 = np.zeros((), dtype=np.float32)
    input_dict_7 = {
        'input': input_7,
        'ord': 'fro',
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_7,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: ord='nuc', complex128 input batch, keepdim=True
    input_8 = (np.random.rand(3, 2, 5) + 1j * np.random.rand(3, 2, 5)).astype(np.complex128)
    out_8 = np.zeros((3, 1, 1), dtype=np.float64)
    input_dict_8 = {
        'input': input_8,
        'ord': 'nuc',
        'dim': (1, 2),
        'keepdim': True,
        'out': out_8,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: ord='fro', negative values
    input_9 = np.array([[-1., -2.], [-3., -4.]], dtype=np.float32)
    out_9 = np.zeros((), dtype=np.float32)
    input_dict_9 = {
        'input': input_9,
        'ord': 'fro',
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_9,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: ord='nuc', non-square matrix
    input_10 = np.random.rand(5, 2).astype(np.float64)
    out_10 = np.zeros((), dtype=np.float64)
    input_dict_10 = {
        'input': input_10,
        'ord': 'nuc',
        'dim': (-2, -1),
        'keepdim': False,
        'out': out_10,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_norm_2"] = linalg_matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.matrix_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_norm_2'.")

check_valid('torch.linalg.matrix_norm', generated_inputs['torch.linalg.matrix_norm_2'], lib="torch", suffix=2)

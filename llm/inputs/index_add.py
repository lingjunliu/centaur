
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def index_add_inputs():
    list_of_inputs = []

    # Case 1: Simple 2D case with float tensor
    input_tensor = np.zeros((3, 5), dtype=np.float32)
    index_tensor = np.array([0, 1, 0], dtype=np.int64)
    source_tensor = np.random.randn(3, 5).astype(np.float32)
    dim = 0
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.zeros((3, 5), dtype=np.float32)
    index_tensor = np.array([0, 1, 0], dtype=np.int64)
    source_tensor = np.random.randn(3, 5).astype(np.float32)
    dim = 0
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.zeros((3, 5), dtype=np.float32)
    index_tensor = np.array([0, 1, 0], dtype=np.int64)
    source_tensor = np.random.randn(3, 5).astype(np.float32)
    dim = 0
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.zeros((3, 5), dtype=np.float32)
    index_tensor = np.array([0, 1, 0], dtype=np.int64)
    source_tensor = np.random.randn(3, 5).astype(np.float32)
    dim = 0
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.zeros((3, 5), dtype=np.float32)
    index_tensor = np.array([0, 1, 0], dtype=np.int64)
    source_tensor = np.random.randn(3, 5).astype(np.float32)
    dim = 0
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = index_add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_add', generated_inputs)

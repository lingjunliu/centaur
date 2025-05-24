
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def chain_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    matrices = [np.random.randn(2, 3).astype(np.float32), np.random.randn(3, 4).astype(np.float32), np.random.randn(4, 2).astype(np.float32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensors
    matrices = [np.random.randint(1, 5, size=(2, 3)).astype(np.int32), np.random.randint(1, 5, size=(3, 4)).astype(np.int32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes
    matrices = [np.random.randn(5, 2).astype(np.float64), np.random.randn(2, 6).astype(np.float64), np.random.randn(6, 3).astype(np.float64), np.random.randn(3, 1).astype(np.float64)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single matrix
    matrices = [np.random.randn(4, 4).astype(np.float32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    matrices = [np.random.randn(2, 3).astype(np.float32) * -1, np.random.randn(3, 2).astype(np.float32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: More matrices with different shapes
    matrices = [np.random.randn(1, 5).astype(np.float32), np.random.randn(5, 3).astype(np.float32), np.random.randn(3, 4).astype(np.float32), np.random.randn(4, 2).astype(np.float32), np.random.randn(2, 1).astype(np.float32)]
    input_dict = {"matrices": matrices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = chain_matmul_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('chain_matmul', generated_inputs)

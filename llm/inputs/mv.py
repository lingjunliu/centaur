
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def mv_inputs():
    list_of_inputs = []

    input_matrix = torch.randn(3, 3).numpy()
    vec = torch.randn(3).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_matrix = torch.randn(5, 5).numpy()
    vec = torch.randn(5).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_matrix = torch.randn(2, 4).numpy()
    vec = torch.randn(4).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_matrix = torch.randn(4, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_matrix = torch.randn(1, 1).numpy()
    vec = torch.randn(1).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = mv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mv', list_of_inputs)

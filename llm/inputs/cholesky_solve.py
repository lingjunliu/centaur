
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cholesky_solve_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(3, 3).numpy()
    input2 = torch.randn(3, 1).numpy()
    input_dict = {"input": input2, "L": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input1 = torch.randn(4, 4).numpy()
    input2 = torch.randn(4, 2).numpy()
    input_dict = {"input": input2, "L": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input1 = torch.randn(5, 5).numpy()
    input2 = torch.randn(5, 3).numpy()
    input_dict = {"input": input2, "L": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input1 = torch.randn(2, 2).numpy()
    input2 = torch.randn(2, 1).numpy()
    input_dict = {"input": input2, "L": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input1 = torch.randn(6, 6).numpy()
    input2 = torch.randn(6, 4).numpy()
    input_dict = {"input": input2, "L": input1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = cholesky_solve_inputs()


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky_solve', list_of_inputs)

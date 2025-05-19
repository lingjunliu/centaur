
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def index_select_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, positive indices
    input1 = torch.randn(3, 4).numpy()
    index1 = torch.tensor([0, 2]).long().numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1, "index": index1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, positive indices
    input2 = torch.randint(0, 10, (2, 3, 5)).int().numpy()
    index2 = torch.tensor([1, 0]).long().numpy()
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2, "index": index2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D complex tensor, positive indices
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    index3 = torch.tensor([0]).long().numpy()
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3, "index": index3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = index_select_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_select', generated_inputs)

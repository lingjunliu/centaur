
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def svdvals_inputs():
    list_of_inputs = []

    A = torch.randn(5, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(3, 5).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(2, 5, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(2, 3, 5).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(1, 2, 5, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(5, 3, dtype=torch.float64).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(3, 5, dtype=torch.float64).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(5, 3, dtype=torch.complex64).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(3, 5, dtype=torch.complex128).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = (torch.randn(5, 3) - 1).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(4, 4)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.random.rand(3, 2, 4, 4)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = svdvals_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('svdvals', generated_inputs)

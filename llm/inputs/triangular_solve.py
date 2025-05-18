
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def triangular_solve_inputs():
    list_of_inputs = []

    A = torch.randn(3, 3).numpy()
    b = torch.randn(3, 1).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(4, 4).numpy()
    b = torch.randn(4, 2).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": False,
        "transpose": True,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(5, 5).numpy()
    b = torch.randn(5, 1).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": True,
        "transpose": True,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(2, 2).numpy()
    b = torch.randn(2, 3).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": False,
        "transpose": False,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(6, 6).numpy()
    b = torch.randn(6, 4).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": True,
        "transpose": False,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = triangular_solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('triangular_solve', list_of_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addmv_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5).numpy()
    mat = torch.randn(5, 4).numpy()
    vec = torch.randn(4).numpy()
    beta = 0.5
    alpha = 2.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.zeros(4).numpy()
    mat = torch.ones(4, 3).numpy()
    vec = torch.ones(3).numpy()
    beta = 0.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, dtype=torch.float64).numpy()
    mat = torch.randn(2, 3, dtype=torch.float64).numpy()
    vec = torch.randn(3, dtype=torch.float64).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))


    input_tensor = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    beta = -0.5
    alpha = -1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = addmv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addmv_', generated_inputs)

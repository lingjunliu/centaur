
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def baddbmm_inputs():
    list_of_inputs = []

    # Input 1
    input_np = torch.randn(3, 5).numpy()
    batch1_np = torch.randn(3, 2, 3).numpy()
    batch2_np = torch.randn(3, 3, 5).numpy()
    beta_val = 1.0
    alpha_val = 2.0
    
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta_val,
        "alpha": alpha_val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_np = torch.randn(2, 5).numpy()
    batch1_np = torch.randn(2, 3, 2).numpy()
    batch2_np = torch.randn(2, 2, 5).numpy()
    beta_val = 0.5
    alpha_val = 1.5
    
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta_val,
        "alpha": alpha_val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_np = torch.randn(4, 6).numpy()
    batch1_np = torch.randn(4, 5, 4).numpy()
    batch2_np = torch.randn(4, 4, 6).numpy()
    beta_val = 0.0
    alpha_val = 1.0
    
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta_val,
        "alpha": alpha_val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_np = torch.randn(1, 1).numpy()
    batch1_np = torch.randn(1, 1, 1).numpy()
    batch2_np = torch.randn(1, 1, 1).numpy()
    beta_val = 2.0
    alpha_val = 0.5
    
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta_val,
        "alpha": alpha_val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_np = torch.randn(5, 7).numpy()
    batch1_np = torch.randn(5, 6, 5).numpy()
    batch2_np = torch.randn(5, 5, 7).numpy()
    beta_val = 1.2
    alpha_val = 0.8
    
    input_dict = {
        "input": input_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "beta": beta_val,
        "alpha": alpha_val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = baddbmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('baddbmm', list_of_inputs)

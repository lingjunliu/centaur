
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    input_dict = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "target": np.random.randint(0, 5, size=(2, 3)).astype(np.float32),
        "log_input": False,
        "full": False,
        "size_average": False,
        "eps": 1e-08,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.random.randn(1, 1).astype(np.float32),
        "target": np.random.randint(0, 5, size=(1, 1)).astype(np.float32),
        "log_input": True,
        "full": True,
        "size_average": False,
        "eps": 1e-06,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.random.randn(4, 2, 2).astype(np.float32),
        "target": np.random.randint(0, 5, size=(4, 2, 2)).astype(np.float32),
        "log_input": False,
        "full": True,
        "size_average": False,
        "eps": 1e-10,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.random.randn(3, 4).astype(np.float32),
        "target": np.random.randint(0, 5, size=(3, 4)).astype(np.float32),
        "log_input": True,
        "full": False,
        "size_average": False,
        "eps": 1e-04,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "input": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "target": np.random.randint(0, 5, size=(2, 2, 2, 2)).astype(np.float32),
        "log_input": False,
        "full": True,
        "size_average": False,
        "eps": 1e-08,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('poisson_nll_loss', list_of_inputs)

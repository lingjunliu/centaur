
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def gaussian_nllloss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with mean reduction
    input1 = torch.randn(5, 2).numpy()
    target1 = torch.randn(5, 2).numpy()
    var1 = torch.ones(5, 2).numpy()
    input_dict1 = {
        "input": input1,
        "target": target1,
        "var": var1,
        "full": False,
        "eps": 1e-6,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Sum reduction with different eps
    input2 = torch.randn(3, 4).numpy()
    target2 = torch.randn(3, 4).numpy()
    var2 = torch.rand(3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "target": target2,
        "var": var2,
        "full": True,
        "eps": 1e-4,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: No reduction
    input3 = torch.randn(2, 2, 2).numpy()
    target3 = torch.randn(2, 2, 2).numpy()
    var3 = torch.rand(2, 2, 2).numpy()
    input_dict3 = {
        "input": input3,
        "target": target3,
        "var": var3,
        "full": False,
        "eps": 1e-6,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Broadcasting var
    input4 = torch.randn(4, 5).numpy()
    target4 = torch.randn(4, 5).numpy()
    var4 = torch.ones(4, 1).numpy()
    input_dict4 = {
        "input": input4,
        "target": target4,
        "var": var4,
        "full": True,
        "eps": 1e-5,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Scalar var
    input5 = torch.randn(10).numpy()
    target5 = torch.randn(10).numpy()
    var5 = np.array(2.0)
    input_dict5 = {
        "input": input5,
        "target": target5,
        "var": var5,
        "full": False,
        "eps": 1e-7,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs = gaussian_nllloss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('GaussianNLLLoss', generated_inputs)

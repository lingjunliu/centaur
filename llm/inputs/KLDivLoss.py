
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def KLDivLoss_inputs():
    generated_inputs = []

    # Case 1: Basic case with default parameters
    input1 = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target1 = torch.rand(3, 5).softmax(dim=1).numpy()
    input_dict1 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False
    }
    generated_inputs.append({"input": input1, "target": target1, **input_dict1})

    # Case 2: Different reduction method "batchmean"
    input2 = torch.randn(2, 4, 6).log_softmax(dim=1).numpy()
    target2 = torch.rand(2, 4, 6).softmax(dim=1).numpy()
    input_dict2 = {
        "size_average": None,
        "reduce": None,
        "reduction": "batchmean",
        "log_target": False
    }
    generated_inputs.append({"input": input2, "target": target2, **input_dict2})

    # Case 3: Different reduction method "sum"
    input3 = torch.randn(1, 3, 3, 3).log_softmax(dim=1).numpy()
    target3 = torch.rand(1, 3, 3, 3).softmax(dim=1).numpy()
    input_dict3 = {
        "size_average": None,
        "reduce": None,
        "reduction": "sum",
        "log_target": False
    }
    generated_inputs.append({"input": input3, "target": target3, **input_dict3})

    # Case 4: log_target = True
    input4 = torch.randn(4, 2).log_softmax(dim=1).numpy()
    target4 = torch.rand(4, 2).log_softmax(dim=1).numpy()
    input_dict4 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": True
    }
    generated_inputs.append({"input": input4, "target": target4, **input_dict4})

    # Case 5: reduction = "none"
    input5 = torch.randn(2, 2).log_softmax(dim=1).numpy()
    target5 = torch.rand(2, 2).softmax(dim=1).numpy()
    input_dict5 = {
        "size_average": None,
        "reduce": None,
        "reduction": "none",
        "log_target": False
    }
    generated_inputs.append({"input": input5, "target": target5, **input_dict5})
    
    # Case 6: 1D tensor
    input6 = torch.randn(5).log_softmax(dim=0).numpy()
    target6 = torch.rand(5).softmax(dim=0).numpy()
    input_dict6 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False
    }
    generated_inputs.append({"input": input6, "target": target6, **input_dict6})

    # Case 7: Different shape
    input7 = torch.randn(1, 2, 3).log_softmax(dim=1).numpy()
    target7 = torch.rand(1, 2, 3).softmax(dim=1).numpy()
    input_dict7 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False
    }
    generated_inputs.append({"input": input7, "target": target7, **input_dict7})

    return generated_inputs

generated_inputs = KLDivLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('KLDivLoss', generated_inputs)

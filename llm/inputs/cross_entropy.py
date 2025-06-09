
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3).numpy()
    target = torch.randint(0, 3, (2,)).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'sum',
        "label_smoothing": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 10).numpy()
    target = torch.randint(0, 10, (1,)).numpy()
    weight = torch.rand(10).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": 5,
        "reduction": 'mean',
        "label_smoothing": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(5, 2).numpy()
    target = torch.randint(0, 2, (5,)).numpy()
    weight = torch.tensor([0.2, 0.8]).float().numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.05
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 6).numpy()
    target = torch.randint(0, 6, (4,)).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -1,
        "reduction": 'mean',
        "label_smoothing": 0.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = cross_entropy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross_entropy', generated_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def nll_loss_inputs():
    list_of_inputs = []

    # Example 1
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    weight = torch.randn(5).numpy()
    ignore_index = -100
    reduction = 'mean'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    input = torch.randn(2, 10).log_softmax(dim=1).numpy()
    target = torch.randint(0, 10, (2,)).numpy()
    weight = None
    ignore_index = -1
    reduction = 'sum'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    input = torch.randn(5, 3).log_softmax(dim=1).numpy()
    target = torch.randint(0, 3, (5,)).numpy()
    weight = torch.ones(3).numpy()
    ignore_index = 1
    reduction = 'mean'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4
    input = torch.randn(1, 7).log_softmax(dim=1).numpy()
    target = torch.randint(0, 7, (1,)).numpy()
    weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    ignore_index = 6
    reduction = 'mean'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5
    input = torch.randn(4, 4).log_softmax(dim=1).numpy()
    target = torch.randint(0, 4, (4,)).numpy()
    weight = None
    ignore_index = -100
    reduction = 'mean'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = nll_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nll_loss', list_of_inputs)

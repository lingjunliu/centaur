
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def nllloss_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = torch.tensor([1, 0, 4]).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).float().numpy()
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

    input = torch.randn(2, 3).log_softmax(dim=1).numpy()
    target = torch.tensor([0, 2]).numpy()
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

    input = torch.randn(4, 4).log_softmax(dim=1).numpy()
    target = torch.tensor([3, 1, 2, 0]).numpy()
    weight = torch.tensor([0.25, 0.25, 0.25, 0.25]).float().numpy()
    ignore_index = 5
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 10).log_softmax(dim=1).numpy()
    target = torch.tensor([7]).numpy()
    weight = None
    ignore_index = 7
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 2).log_softmax(dim=1).numpy()
    target = torch.tensor([0, 1]).numpy()
    weight = torch.tensor([0.7, 0.3]).float().numpy()
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

list_of_inputs = nllloss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('NLLLoss', list_of_inputs)

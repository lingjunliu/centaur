
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []

    input1 = torch.randn(5).numpy()
    input2 = torch.randn(5).numpy()
    target = torch.randint(0, 2, (5,)).float().numpy() * 2 - 1
    margin = 0.0
    size_average = False
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    target = torch.randint(0, 2, (3, 4)).float().numpy() * 2 - 1
    margin = 0.5
    size_average = False
    reduce = True
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 2, 2).numpy()
    input2 = torch.randn(2, 2, 2).numpy()
    target = torch.randint(0, 2, (2, 2, 2)).float().numpy() * 2 - 1
    margin = 1.0
    size_average = False
    reduce = True
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(10).numpy()
    input2 = torch.randn(10).numpy()
    target = torch.randint(0, 2, (10,)).float().numpy() * 2 - 1
    margin = 0.2
    size_average = False
    reduce = True
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 5, 5).numpy()
    input2 = torch.randn(2, 5, 5).numpy()
    target = torch.randint(0, 2, (2, 5, 5)).float().numpy() * 2 - 1
    margin = 0.7
    size_average = False
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = margin_ranking_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MarginRankingLoss', list_of_inputs)

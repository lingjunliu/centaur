
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5).numpy()
    target = torch.empty(3, 5).random_(0, 2).numpy()
    weight = torch.randn(5).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4).numpy()
    target = torch.empty(2, 4).random_(0, 2).numpy()
    weight = None
    size_average = False
    reduce = True
    reduction = 'sum'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3).numpy()
    target = torch.empty(1, 3).random_(0, 2).numpy()
    weight = torch.randn(3).numpy()
    size_average = True
    reduce = False
    reduction = 'none'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2).numpy()
    target = torch.empty(4, 2).random_(0, 2).numpy()
    weight = None
    size_average = False
    reduce = False
    reduction = 'mean'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 6).numpy()
    target = torch.empty(5, 6).random_(0, 2).numpy()
    weight = torch.randn(6).numpy()
    size_average = True
    reduce = True
    reduction = 'sum'

    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = multilabel_soft_margin_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MultiLabelSoftMarginLoss', list_of_inputs)

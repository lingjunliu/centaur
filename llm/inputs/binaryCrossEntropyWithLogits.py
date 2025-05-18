
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (3,), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(3, requires_grad=False).numpy()
    pos_weight_tensor = torch.randn(1, requires_grad=False).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 5, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (2, 5), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(2, 5, requires_grad=False).numpy()
    pos_weight_tensor = torch.randn(1, requires_grad=False).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'sum',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 4, 4, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (1, 4, 4), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(1, 4, 4, requires_grad=False).numpy()
    pos_weight_tensor = torch.randn(1, requires_grad=False).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (5,), dtype=torch.float32).numpy()
    weight_tensor = None
    pos_weight_tensor = None

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4, 5, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (2, 3, 4, 5), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(2, 3, 4, 5, requires_grad=False).numpy()
    pos_weight_tensor = torch.tensor([2.0], requires_grad=False).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = binary_cross_entropy_with_logits_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('binaryCrossEntropyWithLogits', list_of_inputs)

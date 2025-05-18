
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np
import torch.nn.functional as F

def l1_loss_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5)
    target_tensor = torch.randn(3, 5)
    reduction_type = 'mean'

    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4, 6)
    target_tensor = torch.randn(2, 4, 6)
    reduction_type = 'sum'

    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 3, 2, 2)
    target_tensor = torch.randn(1, 3, 2, 2)
    reduction_type = 'none'

    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(size=(4,))
    target_tensor = torch.randn(size=(4,))
    reduction_type = 'mean'

    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2)
    target_tensor = torch.zeros(2, 2)
    reduction_type = 'sum'

    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "reduction": reduction_type
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = l1_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('l1_loss', list_of_inputs)

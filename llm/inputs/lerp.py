
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lerp_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 4).numpy()
    end_tensor = torch.randn(3, 4).numpy()
    weight_tensor = torch.rand(1).numpy()

    input_dict = {
        "input": input_tensor,
        "end": end_tensor,
        "weight": weight_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2, 2).numpy()
    end_tensor = torch.randn(2, 2, 2).numpy()
    weight_tensor = torch.rand(2, 2, 2).numpy()

    input_dict = {
        "input": input_tensor,
        "end": end_tensor,
        "weight": weight_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(5).numpy()
    end_tensor = torch.randn(5).numpy()
    weight_tensor = np.array(0.7)

    input_dict = {
        "input": input_tensor,
        "end": end_tensor,
        "weight": weight_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 5).numpy()
    end_tensor = torch.randn(1, 5, 5).numpy()
    weight_tensor = torch.rand(5).numpy()
    input_dict = {
        "input": input_tensor,
        "end": end_tensor,
        "weight": weight_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4).numpy()
    end_tensor = torch.randn(2, 3, 4).numpy()
    weight_tensor = np.array(0.3).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "end": end_tensor,
        "weight": weight_tensor
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = lerp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lerp', list_of_inputs)

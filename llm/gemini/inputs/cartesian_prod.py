
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cartesian_prod_inputs():
    generated_inputs = []

    # Input 1: Basic integer tensors
    tensors1 = [torch.tensor([1, 2, 3]).numpy(), torch.tensor([4, 5]).numpy()]
    input_dict1 = {"tensors": tensors1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensors
    tensors2 = [torch.tensor([1.1, 2.2]).numpy(), torch.tensor([3.3, 4.4, 5.5]).numpy()]
    input_dict2 = {"tensors": tensors2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Mixed integer and float tensors (make both float)
    tensors3 = [torch.tensor([1.0, 2.0]).numpy(), torch.tensor([3.0, 4.0]).numpy()]
    input_dict3 = {"tensors": tensors3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensors with negative values
    tensors4 = [torch.tensor([-1, 2]).numpy(), torch.tensor([3, -4]).numpy()]
    input_dict4 = {"tensors": tensors4}
    generated_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: More than two tensors
    tensors5 = [torch.tensor([1, 2]).numpy(), torch.tensor([3, 4]).numpy(), torch.tensor([5]).numpy()]
    input_dict5 = {"tensors": tensors5}
    generated_inputs.append(copy.deepcopy(input_dict5))

    return generated_inputs

generated_inputs = cartesian_prod_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cartesian_prod', generated_inputs)

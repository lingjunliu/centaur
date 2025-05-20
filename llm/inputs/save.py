
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy
import io

def torch_save_inputs():
    list_of_inputs = []

    # Input 1: Simple float tensor
    x = torch.tensor([0.1, 1.2, 2.3, 3.4, 4.5]).numpy()
    f = "tensor1.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor with negative values
    x = torch.tensor([-1, 0, 1, 2, -3]).numpy()
    f = "tensor2.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    x = torch.randn(2, 3).numpy()
    f = "tensor3.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    x = torch.randn(2, 3, 4).numpy()
    f = "tensor4.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex tensor
    x = torch.randn(2, 2, dtype=torch.complex64).numpy()
    f = "tensor5.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Bool tensor
    x = torch.tensor([True, False, True]).numpy()
    f = "tensor6.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero dimensional tensor (scalar)
    x = torch.tensor(5.0).numpy()
    f = "tensor7.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = torch_save_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('save', generated_inputs)

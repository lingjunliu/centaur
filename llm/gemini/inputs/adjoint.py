
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def adjoint_inputs():
    generated_inputs = []

    # Real valued tensor, 2D
    input1 = torch.randn(2, 3).numpy()
    generated_inputs.append({"input": input1})

    # Real valued tensor, 3D
    input2 = torch.randn(2, 3, 4).numpy()
    generated_inputs.append({"input": input2})

    # Real valued tensor, 4D
    input3 = torch.randn(2, 3, 4, 5).numpy()
    generated_inputs.append({"input": input3})

    # Complex valued tensor, 2D
    input4 = torch.randn(2, 3, dtype=torch.float)
    input4 = torch.complex(input4, torch.randn(2, 3, dtype=torch.float)).numpy()
    generated_inputs.append({"input": input4})

    # Complex valued tensor, 3D
    input5 = torch.randn(2, 3, 4, dtype=torch.float)
    input5 = torch.complex(input5, torch.randn(2, 3, 4, dtype=torch.float)).numpy()
    generated_inputs.append({"input": input5})

    # Integer tensor, 2D
    input6 = torch.randint(-10, 10, (2, 3)).numpy()
    generated_inputs.append({"input": input6})

    # Integer tensor, 3D
    input7 = torch.randint(-10, 10, (2, 3, 4)).numpy()
    generated_inputs.append({"input": input7})

    # Complex valued tensor, 4D
    input8 = torch.randn(2, 3, 4, 5, dtype=torch.float)
    input8 = torch.complex(input8, torch.randn(2, 3, 4, 5, dtype=torch.float)).numpy()
    generated_inputs.append({"input": input8})

    # Real valued tensor with negative values
    input9 = torch.randn(5, 5).numpy() * -1
    generated_inputs.append({"input": input9})

    return generated_inputs

generated_inputs = adjoint_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('adjoint', generated_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def identity_inputs():
    generated_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    generated_inputs.append({"input": input1})

    # Input 2: 2D int tensor
    input2 = torch.randint(-10, 10, (3, 4)).numpy().astype(np.int64)
    generated_inputs.append({"input": input2})

    # Input 3: 3D float tensor (complex causing issues)
    input3 = torch.randn(2, 3, 4).numpy()
    generated_inputs.append({"input": input3})

    # Input 4: 4D bool tensor
    input4 = torch.randint(0, 2, (2, 2, 2, 2)).bool().numpy()
    generated_inputs.append({"input": input4})

    # Input 5: 5D float tensor with negative values
    input5 = torch.randn(1, 2, 3, 4, 5).numpy()
    generated_inputs.append({"input": input5})

    return generated_inputs

generated_inputs = identity_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Identity', generated_inputs)

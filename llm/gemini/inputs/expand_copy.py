
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def expand_copy_inputs():
    list_of_inputs = []

    # Case 1: 1D int tensor
    input1 = torch.randint(0, 10, (3,)).numpy()
    size1 = (2, 3)
    input_dict1 = {"input": input1, "size": size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = expand_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('expand_copy', generated_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_diagonal_inputs():
    list_of_inputs = []

    input_1 = torch.randn(3, 3).numpy()
    input_dict_1 = {"input": input_1, "offset": 0, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.diagonal"] = torch_diagonal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('diagonal', generated_inputs['torch.diagonal'], lib="torch")


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def isclose_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    rtol1 = 1e-05
    atol1 = 1e-08
    equal_nan1 = False
    input_dict1 = {
        "input": input1,
        "other": other1,
        "rtol": rtol1,
        "atol": atol1,
        "equal_nan": equal_nan1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([float('nan'), 1.0, 2.0]).numpy()
    other2 = torch.tensor([float('nan'), 1.0, 2.0]).numpy()
    rtol2 = 1e-05
    atol2 = 1e-08
    equal_nan2 = True
    input_dict2 = {
        "input": input2,
        "other": other2,
        "rtol": rtol2,
        "atol": atol2,
        "equal_nan": equal_nan2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other3 = torch.tensor([1.0 + 1e-7, 2.0, 3.0 - 1e-7]).numpy()
    rtol3 = 1e-06
    atol3 = 0.0
    equal_nan3 = False
    input_dict3 = {
        "input": input3,
        "other": other3,
        "rtol": rtol3,
        "atol": atol3,
        "equal_nan": equal_nan3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other4 = torch.tensor([1.0, 2.0 + 1e-9, 3.0]).numpy()
    rtol4 = 0.0
    atol4 = 1e-08
    equal_nan4 = False
    input_dict4 = {
        "input": input4,
        "other": other4,
        "rtol": rtol4,
        "atol": atol4,
        "equal_nan": equal_nan4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, dtype=torch.float64).numpy()
    other5 = torch.randn(2, 2, dtype=torch.float64).numpy()
    rtol5 = 1e-08
    atol5 = 1e-05
    equal_nan5 = False

    input_dict5 = {
        "input": input5,
        "other": other5,
        "rtol": rtol5,
        "atol": atol5,
        "equal_nan": equal_nan5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = isclose_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('isclose', list_of_inputs)


from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bitwise_or_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    other1 = torch.tensor([4, 3, 2, 1], dtype=torch.int32).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8).numpy()
    other2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int8).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([True, False, True, False]).numpy()
    other3 = torch.tensor([False, True, False, True]).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    other4 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1, 2, 3, 4], dtype=torch.uint8).numpy()
    other5 = torch.tensor([4, 3, 2, 1], dtype=torch.uint8).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = bitwise_or_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bitwise_or', list_of_inputs)

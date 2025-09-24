
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def take_inputs():
    list_of_inputs = []

    src = torch.tensor([[4, 3, 5], [6, 7, 8]])
    index = torch.tensor([0, 2, 5])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    src = torch.tensor([[1.1, 2.2], [3.3, 4.4]])
    index = torch.tensor([0, 1, 2, 3])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    src = torch.arange(12).reshape(3, 4)
    index = torch.tensor([0, 5, 11, 7])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    src = torch.randn(2, 3, 4)
    index = torch.randint(0, 24, (5,))
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    src = torch.randint(-10, 10, (5, 5))
    index = torch.tensor([0, -1, 5, -5])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    src = torch.tensor([1, 2, 3, 4, 5])
    index = torch.tensor([0, 0, 1, 2, 2, 3])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    src = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
    index = torch.tensor([0, 2, 4])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = take_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('take', generated_inputs)

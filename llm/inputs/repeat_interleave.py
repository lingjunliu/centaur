
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def repeat_interleave_inputs():
    list_of_inputs = []

    input_tensor = torch.tensor([1, 2, 3]).numpy()
    repeats_tensor = 2
    input_dict = {"input": input_tensor, "repeats": repeats_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3).numpy()
    repeats_tensor = torch.tensor([1, 2])
    input_dict = {"input": input_tensor, "repeats": repeats_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    repeats_tensor = torch.tensor([2, 3])
    input_dict = {"input": input_tensor, "repeats": repeats_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1, 2, 3, 4]).numpy()
    repeats_tensor = 3
    input_dict = {"input": input_tensor, "repeats": repeats_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5).numpy()
    repeats_tensor = torch.tensor([4])
    input_dict = {"input": input_tensor, "repeats": repeats_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = repeat_interleave_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('repeat_interleave', list_of_inputs)

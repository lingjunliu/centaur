
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ldexp_inputs():
    list_of_inputs = []

    input_tensor = torch.tensor([1.0]).numpy()
    other_tensor = torch.tensor([1]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([1.0]).numpy()
    other_tensor = torch.tensor([1, 2, 3, 4]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other_tensor = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3).numpy()
    other_tensor = torch.randint(1, 5, (2, 3)).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = ldexp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ldexp', generated_inputs)

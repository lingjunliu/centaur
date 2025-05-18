
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def heaviside_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 4).numpy()
    values_tensor = torch.tensor(0.5).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2, 2).numpy()
    values_tensor = torch.tensor(1.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5).numpy()
    values_tensor = torch.tensor(0.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randint(-5, 5, (2, 3)).float().numpy()
    values_tensor = torch.tensor(2.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 5, 5).numpy()
    values_tensor = torch.tensor(-1.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = heaviside_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('heaviside', list_of_inputs)

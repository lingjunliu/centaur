
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lerp_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors with scalar weight
    start = torch.arange(1., 5.)
    end = torch.full_like(torch.arange(1., 5.), 10.)
    weight = 0.5
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Float tensors with tensor weight
    start = torch.arange(1., 5.)
    end = torch.full_like(torch.arange(1., 5.), 10.)
    weight = torch.full_like(torch.arange(1., 5.), 0.5)
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Int tensors with scalar weight
    start = torch.arange(1, 5).float()
    end = torch.full_like(torch.arange(1, 5).float(), 10.)
    weight = 0.5
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Int tensors with tensor weight
    start = torch.arange(1, 5).float()
    end = torch.full_like(torch.arange(1, 5).float(), 10.)
    weight = torch.full_like(torch.arange(1, 5, dtype=torch.float32), 0.5)
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Negative values and different weight
    start = torch.arange(-2., 2.)
    end = torch.full_like(torch.arange(-2., 2.), -5.)
    weight = 0.75
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lerp_inputs()

for i in range(len(generated_inputs)):
    generated_inputs[i]["input"] = torch.from_numpy(generated_inputs[i]["input"])
    generated_inputs[i]["end"] = torch.from_numpy(generated_inputs[i]["end"])
    if type(generated_inputs[i]["weight"]) is not float:
        generated_inputs[i]["weight"] = torch.from_numpy(generated_inputs[i]["weight"])

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lerp', generated_inputs)

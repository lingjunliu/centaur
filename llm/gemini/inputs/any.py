
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def any_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3).bool().numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randint(0, 2, (4, 4)).bool().numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5) < 0.5
    input = input.numpy()
    dim = 2
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 2, 4) > 0
    input = input.numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = (torch.arange(12).reshape(3, 4) % 2).bool().numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[True, False], [False, True]], [[False, False], [True, True]]]).numpy()
    dim = (0, 1)
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[True, False], [False, True]], [[False, False], [True, True]]]).numpy()
    dim = (0, 2)
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = any_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('any', generated_inputs)

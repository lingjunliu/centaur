
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def gather_inputs():
    list_of_inputs = []

    # Case 1: 2D tensor, dim=0
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    index = torch.tensor([[0, 0], [1, 0]]).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D tensor, dim=1
    input = torch.randn(2, 3, 4).numpy()
    index = torch.randint(0, 3, (2, 2, 4)).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D tensor, dim=1, negative indices - REMOVED as it was causing errors
    # input = torch.randn(3, 5).numpy()
    # index = torch.tensor([[0, -1, 2], [1, -2, 3], [2, -3, 4]]).long().numpy()
    # dim = 1
    # input_dict = {"input": input, "dim": dim, "index": index}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D tensor, dim=0
    input = torch.arange(5).float().numpy()
    index = torch.tensor([0, 2, 4]).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 4D tensor, dim=2
    input = torch.randn(1, 2, 3, 4).numpy()
    index = torch.randint(0, 3, (1, 2, 2, 4)).long().numpy()
    dim = 2
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 2D int tensor
    input = torch.randint(0, 10, (2, 3)).int().numpy()
    index = torch.tensor([[0, 1], [2, 0]]).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 3D tensor, dim=0
    input = torch.randn(3, 4, 5).numpy()
    index = torch.randint(0, 3, (2, 4, 5)).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: 2D tensor, dim=-1 (last dimension) - Modified to prevent index out of bound
    input = torch.randn(3, 5).numpy()
    index = torch.randint(0, 5, (3, 2)).long().numpy() # Ensure index < 5
    dim = -1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = gather_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('gather', generated_inputs)

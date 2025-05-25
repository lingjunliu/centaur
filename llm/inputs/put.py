
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def put_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with 1D tensors and accumulate=False
    input_tensor = torch.randn(5).numpy()
    index_tensor = torch.tensor([0, 2, 4]).numpy()
    source_tensor = torch.tensor([10, 20, 30]).numpy()
    accumulate = False
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 2: Int type, accumulate=False, different shape
    input_tensor = torch.randint(0, 10, (2, 2)).numpy()
    index_tensor = torch.tensor([[0, 0], [1, 1]]).numpy()
    source_tensor = torch.tensor([100, 200]).numpy()
    accumulate = False
    input_dict = {"input": input_tensor, "index": index_tensor, "source": source_tensor, "accumulate": accumulate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = put_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('put', generated_inputs)

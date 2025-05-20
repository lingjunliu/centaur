
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([3.0, 1.0, 2.0]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Int tensors
    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([3, 1, 2]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and different shapes (broadcasting)
    input1 = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    input2 = torch.tensor([[-2.0], [1.0], [-4.0]]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NaNs
    input1 = torch.tensor([1.0, float('nan'), 3.0]).numpy()
    input2 = torch.tensor([float('nan'), 2.0, float('nan')]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Both NaNs
    input1 = torch.tensor([float('nan'), float('nan')]).numpy()
    input2 = torch.tensor([float('nan'), float('nan')]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Multi-dimensional tensors
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(2, 3).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Broadcasting with multi-dimensional tensors
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(3).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: Integer and Float tensors with different shapes
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([1.5, 3.5]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 9: Larger Tensors
    input1 = torch.randn(5, 5, 5).numpy()
    input2 = torch.randn(5, 5, 5).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = fmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fmax', generated_inputs)

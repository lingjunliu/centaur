
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with float tensors and dim=1
    a = torch.randn(4, 3).numpy()
    b = torch.randn(4, 3).numpy()
    input_dict = {"input": a, "other": b, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Basic case with float tensors and no dim specified
    a = torch.randn(4, 3).numpy()
    b = torch.randn(4, 3).numpy()
    input_dict = {"input": a, "other": b, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Integer tensors with dim=0
    a = torch.randint(-5, 5, (3, 3)).numpy()
    b = torch.randint(-5, 5, (3, 3)).numpy()
    input_dict = {"input": a, "other": b, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Complex tensors with dim=-1 (should fail in torch.cross but test valid input format)
    a = torch.randn(3, 3, dtype=torch.complex64).numpy()
    b = torch.randn(3, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": a, "other": b, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Different sized tensors (should cause a broadcast error when using torch.cross)
    a = torch.randn(3).numpy()
    b = torch.randn(3).numpy()
    input_dict = {"input": a, "other": b, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: 3D tensors with dim=2
    a = torch.randn(2, 4, 3).numpy()
    b = torch.randn(2, 4, 3).numpy()
    input_dict = {"input": a, "other": b, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Double tensors
    a = torch.randn(4, 3, dtype=torch.float64).numpy()
    b = torch.randn(4, 3, dtype=torch.float64).numpy()
    input_dict = {"input": a, "other": b, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cross"] = cross_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross', generated_inputs['torch.cross'], lib="torch")

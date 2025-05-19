
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cat_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 2D float tensors
    tensors1 = [torch.randn(2, 3).numpy(), torch.randn(2, 3).numpy()]
    dim1 = 0
    input_dict1 = {"tensors": tensors1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Three 1D int tensors, concatenating along the first dimension
    tensors2 = [torch.randint(0, 10, (5,)).numpy(), torch.randint(0, 10, (5,)).numpy(), torch.randint(0, 10, (5,)).numpy()]
    dim2 = 0
    input_dict2 = {"tensors": tensors2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Two 3D tensors with different shapes, concatenating along dim=1
    tensors3 = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 5, 4).numpy()]
    dim3 = 1
    input_dict3 = {"tensors": tensors3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: One tensor in the list
    tensors4 = [torch.randn(2, 3).numpy()]
    dim4 = 0
    input_dict4 = {"tensors": tensors4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Two 2D complex tensors
    tensors5 = [torch.randn(2, 3, dtype=torch.complex64).numpy(), torch.randn(2, 3, dtype=torch.complex64).numpy()]
    dim5 = 1
    input_dict5 = {"tensors": tensors5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Two tensors, negative dim
    tensors6 = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy()]
    dim6 = -1
    input_dict6 = {"tensors": tensors6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 4D tensors
    tensors7 = [torch.randn(2,3,4,5).numpy(), torch.randn(2,3,4,5).numpy()]
    dim7 = 2
    input_dict7 = {"tensors": tensors7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = cat_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cat', generated_inputs)

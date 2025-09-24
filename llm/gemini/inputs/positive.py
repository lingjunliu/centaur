
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def positive_inputs():
    list_of_inputs = []

    # Float tensor
    input_float = torch.randn(5).numpy()
    input_dict = {"input": input_float}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Int tensor
    input_int = torch.randint(-10, 10, (3, 3)).numpy()
    input_dict = {"input": input_int}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Complex tensor
    input_complex = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict = {"input": input_complex}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Multi-dimensional tensor
    input_multidim = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_multidim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Tensor with negative values
    input_negative = torch.randn(4, 4) * -1.0
    input_negative = input_negative.numpy()
    input_dict = {"input": input_negative}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Scalar tensor
    input_scalar = torch.tensor(5.0).numpy()
    input_dict = {"input": input_scalar}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Zero tensor
    input_zero = torch.zeros(2, 2).numpy()
    input_dict = {"input": input_zero}
    list_of_inputs.append(copy.deepcopy(input_dict))

    #Tensor with a mix of positive and negative values
    input_mixed = torch.tensor([-1.0, 2.0, -3.0, 4.0]).numpy()
    input_dict = {"input": input_mixed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = positive_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('positive', generated_inputs)

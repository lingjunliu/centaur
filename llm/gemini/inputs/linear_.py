
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def linear_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    input_1 = torch.randn(3, 4).numpy()
    weight_1 = torch.randn(5, 4).numpy()
    bias_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1, "weight": weight_1, "bias": bias_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Input with batch dimension
    input_2 = torch.randn(2, 3, 4).numpy()
    weight_2 = torch.randn(5, 4).numpy()
    bias_2 = torch.randn(5).numpy()
    input_dict_2 = {"input": input_2, "weight": weight_2, "bias": bias_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Integer tensors
    input_3 = torch.randint(0, 10, (3, 4)).numpy()
    weight_3 = torch.randint(0, 10, (5, 4)).numpy()
    bias_3 = torch.randint(0, 10, (5,)).numpy()
    input_dict_3 = {"input": input_3, "weight": weight_3, "bias": bias_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: No bias
    input_4 = torch.randn(3, 4).numpy()
    weight_4 = torch.randn(5, 4).numpy()
    input_dict_4 = {"input": input_4, "weight": weight_4, "bias": None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Single input vector
    input_5 = torch.randn(4).numpy()
    weight_5 = torch.randn(5, 4).numpy()
    bias_5 = torch.randn(5).numpy()
    input_dict_5 = {"input": input_5, "weight": weight_5, "bias": bias_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Larger input and weight dimensions
    input_6 = torch.randn(10, 20).numpy()
    weight_6 = torch.randn(30, 20).numpy()
    bias_6 = torch.randn(30).numpy()
    input_dict_6 = {"input": input_6, "weight": weight_6, "bias": bias_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Negative Values
    input_7 = torch.randn(3, 4) * -1.0
    weight_7 = torch.randn(5, 4) * -1.0
    bias_7 = torch.randn(5) * -1.0
    input_dict_7 = {"input": input_7.numpy(), "weight": weight_7.numpy(), "bias": bias_7.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs = linear_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('linear_', generated_inputs)

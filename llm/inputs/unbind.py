
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def unbind_inputs():
    list_of_inputs = []

    # Case 1: 2D integer tensor, dim=0
    input1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D float tensor, dim=1
    input2 = torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]).numpy()
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 3D integer tensor, dim=0
    input3 = torch.randint(0, 10, (2, 3, 4)).numpy()
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 3D float tensor, dim=1
    input4 = torch.randn(3, 4, 5).numpy()
    dim4 = 1
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 1D integer tensor, dim=0
    input5 = torch.arange(5).numpy()
    dim5 = 0
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: 2D integer tensor, dim=-1
    input6 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    dim6 = -1
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: 3D integer tensor, dim=-2
    input7 = torch.randint(0, 10, (2, 3, 4)).numpy()
    dim7 = -2
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Case 8: 4D float tensor, dim=2
    input8 = torch.randn(2, 3, 4, 5).numpy()
    dim8 = 2
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Case 9: 4D integer tensor, dim=-1
    input9 = torch.randint(0, 10, (2, 3, 4, 5)).numpy()
    dim9 = -1
    input_dict9 = {"input": input9, "dim": dim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Case 10: Empty tensor
    input10 = torch.tensor([]).numpy()
    dim10 = 0
    input_dict10 = {"input": input10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = unbind_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unbind', generated_inputs)

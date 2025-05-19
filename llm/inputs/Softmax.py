
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input1 = np.array([1.0, 2.0, 3.0])
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with negative values
    input2 = np.array([[-1.0, 0.5, 2.0], [3.0, -2.0, 1.5]])
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = np.random.rand(2, 3, 4)
    dim3 = 2
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor
    input4 = np.random.randn(1, 5, 5, 2)
    dim4 = 3
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: large values
    input5 = np.array([[1000.0, 1001.0, 1002.0], [1003.0, 999.0, 1000.5]])
    dim5 = 1
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: tensor with a different dim value
    input6 = np.random.rand(2, 3, 4)
    dim6 = 0
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Softmax', generated_inputs)

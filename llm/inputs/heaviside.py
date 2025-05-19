
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, value=0.0
    input1 = np.array([-1.0, 0.0, 1.0, 2.5]).astype(np.float32)
    values1 = np.array([0.0]).astype(np.float32)
    input_dict1 = {"input": input1, "values": values1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor, value=1
    input2 = np.array([-2, -1, 0, 1, 2]).astype(np.int32)
    values2 = np.array([1]).astype(np.int32)
    input_dict2 = {"input": input2, "values": values2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional float tensor, value=0.5
    input3 = np.random.randn(2, 3, 4).astype(np.float64)
    values3 = np.array([0.5]).astype(np.float64)
    input_dict3 = {"input": input3, "values": values3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Boolean tensor, value=True (1)
    input4 = np.array([True, False, True, False]).astype(np.bool_)
    values4 = np.array([True]).astype(np.bool_)

    input_dict4 = {"input": input4, "values": values4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Negative value array
    input5 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]).astype(np.float32)
    values5 = np.array([0.0]).astype(np.float32)
    input_dict5 = {"input": input5, "values": values5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Scalar input and value
    input6 = np.array(0.0).astype(np.float32)
    values6 = np.array(1.0).astype(np.float32)
    input_dict6 = {"input": input6, "values": values6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float64 array with a different value
    input7 = np.array([-1.5, -0.5, 0.0, 0.5, 1.5]).astype(np.float64)
    values7 = np.array([1.0]).astype(np.float64)
    input_dict7 = {"input": input7, "values": values7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = heaviside_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('heaviside', generated_inputs)

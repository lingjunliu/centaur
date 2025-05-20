
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def multigammaln_inputs():
    list_of_inputs = []

    input1 = np.random.rand(5).astype(np.float32)
    p1 = 3
    safe1 = True
    input_dict1 = {"input": input1, "p": p1, "safe": safe1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3).astype(np.float64)
    p2 = 2
    safe2 = False
    input_dict2 = {"input": input2, "p": p2, "safe": safe2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = (np.random.rand(3, 2) + 1j*np.random.rand(3, 2)).astype(np.complex64)
    p3 = 1
    safe3 = True
    input_dict3 = {"input": input3, "p": p3, "safe": safe3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = (np.random.rand(4) + 1j*np.random.rand(4)).astype(np.complex128)
    p4 = 4
    safe4 = False
    input_dict4 = {"input": input4, "p": p4, "safe": safe4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.5, 2.5, 3.5, 4.5]).astype(np.float32)
    p5 = 2
    safe5 = True
    input_dict5 = {"input": input5, "p": p5, "safe": safe5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(2, 2, 2).astype(np.float64)
    p6 = 3
    safe6 = False
    input_dict6 = {"input": input6, "p": p6, "safe": safe6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[2.0, 3.0], [4.0, 5.0]]).astype(np.float32)
    p7 = 1
    safe7 = True
    input_dict7 = {"input": input7, "p": p7, "safe": safe7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = multigammaln_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multigammaln', generated_inputs)

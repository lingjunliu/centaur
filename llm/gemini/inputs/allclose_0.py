
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def allclose_inputs():
    list_of_inputs = []

    input1 = np.array([10000., 1e-07])
    input2 = np.array([10000.1, 1e-08])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([10000., 1e-08])
    input2 = np.array([10000.1, 1e-09])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, float('nan')])
    input2 = np.array([1.0, float('nan')])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, float('nan')])
    input2 = np.array([1.0, float('nan')])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[1.0 + 1e-9, 2.0 + 1e-9], [3.0 + 1e-9, 4.0 + 1e-9]])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([-1.0, -2.0, -3.0])
    input2 = np.array([-1.0 - 1e-7, -2.0 - 1e-7, -3.0 - 1e-7])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input2 = np.array([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7], dtype=np.float64)
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.allclose"] = allclose_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.allclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.allclose'.")

check_valid('torch.allclose', generated_inputs['torch.allclose'], lib="torch")

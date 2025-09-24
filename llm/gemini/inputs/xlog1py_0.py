
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def xlog1py_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0])
    y = np.array([4.0, 5.0, 6.0])
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    y = np.array([4.0, 5.0, 6.0]).reshape(1, 3)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    y = np.array([4.0, 5.0, 6.0]).reshape(3, 1)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.xlog1py"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.special.xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py'.")

check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py'], lib="torch")

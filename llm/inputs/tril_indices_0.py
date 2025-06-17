
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def tril_indices_inputs():
    list_of_inputs = []

    input_dict = {
        "row": np.int64(5),
        "col": np.int64(5),
        "offset": np.int64(0),
        "dtype": None,
        "layout": None,
        "device": None,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(3),
        "col": np.int64(7),
        "offset": np.int64(1),
        "dtype": None,
        "layout": None,
        "device": None,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(7),
        "col": np.int64(3),
        "offset": np.int64(-1),
        "dtype": None,
        "layout": None,
        "device": None,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(10),
        "col": np.int64(10),
        "offset": np.int64(3),
        "dtype": None,
        "layout": None,
        "device": None,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "row": np.int64(4),
        "col": np.int64(6),
        "offset": np.int64(-2),
        "dtype": None,
        "layout": None,
        "device": None,
        "pin_memory": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.tril_indices"] = tril_indices_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.tril_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril_indices'.")

check_valid('torch.tril_indices', generated_inputs['torch.tril_indices'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def triu_indices_inputs():
    list_of_inputs = []

    input_dict = {
        "row": np.int64(5),
        "col": np.int64(5),
        "offset": np.int64(0),
        "dtype": torch.int64,
        "layout": torch.strided
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(3),
        "col": np.int64(4),
        "offset": np.int64(1),
        "dtype": torch.int32,
        "layout": torch.strided
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(6),
        "col": np.int64(6),
        "offset": np.int64(-1),
        "dtype": torch.int64,
        "layout": torch.strided
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(7),
        "col": np.int64(2),
        "offset": np.int64(2),
        "dtype": torch.int32,
        "layout": torch.strided
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "row": np.int64(4),
        "col": np.int64(4),
        "offset": np.int64(-2),
        "dtype": torch.int64,
        "layout": torch.strided
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.triu_indices"] = triu_indices_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.triu_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.triu_indices'.")

check_valid('torch.triu_indices', generated_inputs['torch.triu_indices'], lib="torch")

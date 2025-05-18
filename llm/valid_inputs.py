import torch, copy
import numpy as np

def full_inputs():
    list_of_inputs = []

    input_dict = {
        "size": (2, 3),
        "fill_value": 1.0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (5,),
        "fill_value": 2.5,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 2, 2),
        "fill_value": -1.0,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (1, 4, 4),
        "fill_value": 0.0,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (3, 1),
        "fill_value": 100.0,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = full_inputs()



from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def bucketize_inputs():
    list_of_inputs = []

    boundaries = np.array([1, 3, 5, 7, 9])
    v = np.array([[3, 6, 9], [3, 6, 9]])
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": False,
        "right": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    boundaries = np.array([1, 3, 5, 7, 9])
    v = np.array([[3, 6, 9], [3, 6, 9]])
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": False,
        "right": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    boundaries = np.array([1.5, 3.5, 5.5, 7.5, 9.5])
    v = np.array([[1.6, 3.4, 5.7], [2.1, 7.2, 8.9]])
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": True,
        "right": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    boundaries = np.array([-5, -3, -1, 1, 3])
    v = np.array([[-4, -2, 0], [2, 4, 6]])
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": True,
        "right": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    boundaries = np.array([1, 3, 5, 7, 9])
    v = np.array([3, 6, 9])
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": False,
        "right": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    boundaries = np.array([1, 3, 5, 7, 9])
    v = np.array(3.5)
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": False,
        "right": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    boundaries = np.array([1, 3, 5, 7, 9])
    v = np.array([[[3, 6], [9,2]], [[3, 6], [9,6]]])
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": False,
        "right": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    boundaries = np.array([1, 3, 5])
    v = np.array([1,3,5,7,9])
    out = np.zeros_like(v, dtype=np.int64)
    input_dict = {
        "input": v,
        "boundaries": boundaries,
        "out_int32": False,
        "right": True,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.bucketize"] = bucketize_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bucketize'.")

check_valid('torch.bucketize', generated_inputs['torch.bucketize'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def embedding_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64)
    weight1 = np.random.rand(10, 3).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "padding_idx": None,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0, 2, 0, 5]], dtype=np.int64)
    weight2 = np.random.rand(10, 3).astype(np.float32)
    weight2[0, :] = 0
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "padding_idx": 0,
        "max_norm": None,
        "norm_type": 2.0,
        "scale_grad_by_freq": False,
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3], dtype=np.int64)
    weight3 = np.random.rand(5, 2).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "padding_idx": None,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "scale_grad_by_freq": True,
        "sparse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 2, 3], [4, 3, 2]], dtype=np.int64)
    weight4 = np.random.rand(5, 4).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "padding_idx": 0,
        "max_norm": 1.5,
        "norm_type": 3.0,
        "scale_grad_by_freq": True,
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[1, 2], [3, 4]], [[0, 1], [2, 3]]], dtype=np.int64)
    weight5 = np.random.rand(5, 5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "padding_idx": 1,
        "max_norm": 0.8,
        "norm_type": 2.5,
        "scale_grad_by_freq": False,
        "sparse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.embedding"] = embedding_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.embedding' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding'.")

check_valid('torch.nn.functional.embedding', generated_inputs['torch.nn.functional.embedding'], lib="torch")

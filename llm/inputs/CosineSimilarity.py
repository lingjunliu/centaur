
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cosine_similarity_inputs():
    list_of_inputs = []

    input1 = np.random.randn(100, 128).astype(np.float32)
    input2 = np.random.randn(100, 128).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 1,
        "eps": 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(50, 64).astype(np.float32)
    input2 = np.random.randn(50, 64).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 0,
        "eps": 1e-06
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.randn(20, 30, 40).astype(np.float32)
    input2 = np.random.randn(20, 30, 40).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": -1,
        "eps": 1e-04
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(10, 20, 30).astype(np.float32)
    input2 = np.random.randn(10, 20, 30).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 2,
        "eps": 1e-02
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(5, 10).astype(np.float32)
    input2 = np.random.randn(5, 10).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 1,
        "eps": 1e-12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = cosine_similarity_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CosineSimilarity', generated_inputs)

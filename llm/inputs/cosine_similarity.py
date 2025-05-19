
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cosine_similarity_inputs():
    list_of_inputs = []

    x1 = np.random.randn(3, 5).astype(np.float32)
    x2 = np.random.randn(3, 5).astype(np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(10).astype(np.float64)
    x2 = np.random.randn(10).astype(np.float64)
    dim = 0
    eps = 1e-6
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    dim = 2
    eps = 1e-12
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.random.randn(4, 4).astype(np.float32)
    x2 = np.random.randn(4, 4).astype(np.float32)
    dim = 0
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.random.randn(5, 2).astype(np.float32)
    x2 = np.random.randn(5, 2).astype(np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
    x2 = np.array([[7,8,9],[10,11,12]], dtype=np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
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

check_valid('cosine_similarity', generated_inputs)

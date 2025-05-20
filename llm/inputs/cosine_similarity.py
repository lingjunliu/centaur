
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def cosine_similarity_inputs():
    list_of_inputs = []

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    dim = 1
    eps = 1e-8

    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    x2 = np.array([[7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]])
    dim = 1
    eps = 1e-8

    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    dim = 0
    eps = 1e-8

    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    x2 = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    dim = 2
    eps = 1e-8

    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64)
    dim = 1
    eps = 1e-8

    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
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

check_valid('cosine_similarity', generated_inputs)

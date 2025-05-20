
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def hinge_embedding_loss_inputs():
    list_of_inputs = []

    input1 = np.array([0.5, 0.8, 0.2, 0.9])
    target1 = np.array([1, -1, 1, -1])
    input_dict1 = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append([input1, target1, input_dict1])

    return list_of_inputs

generated_inputs = hinge_embedding_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('HingeEmbeddingLoss', generated_inputs)

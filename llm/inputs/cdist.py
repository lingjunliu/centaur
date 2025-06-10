
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch
import numpy as np
import copy

def cdist_inputs():
    list_of_inputs = []

    # Input 1: Basic case with p=2
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p = 2.0
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes
    x1 = np.array([[1.0, 2.0, 3.0]])
    x2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p = 2.0
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different p value
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p = 1.0
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    x1 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    x2 = np.array([[5.0, -6.0], [-7.0, 8.0]])
    p = 2.0
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: p = infinity
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p = float('inf')
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    x2 = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    p = 2.0
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: p = 0
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p = 0.0
    input_dict = {"x1": x1, "x2": x2, "p": p}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cdist"] = cdist_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cdist', generated_inputs['torch.cdist'], lib="torch")

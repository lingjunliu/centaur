
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cdist_inputs():
    list_of_inputs = []

    # Example 1: Basic example with p=2
    x1 = np.array([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]], dtype=np.float32)
    x2 = np.array([[-2.1763, -0.4713], [-0.6986, 1.3702]], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "compute_mode": "use_mm_for_euclid_dist_if_necessary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different p value (p=1)
    x1 = np.random.rand(5, 3).astype(np.float32)
    x2 = np.random.rand(4, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.0,
        "compute_mode": "use_mm_for_euclid_dist_if_necessary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Different compute_mode
    x1 = np.random.rand(3, 2).astype(np.float32)
    x2 = np.random.rand(2, 2).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "compute_mode": "use_mm_for_euclid_dist"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Negative values and different shapes
    x1 = np.random.randn(4, 5).astype(np.float32)
    x2 = np.random.randn(6, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 3.0,
        "compute_mode": "donot_use_mm_for_euclid_dist"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: p = infinity (np.inf)
    x1 = np.random.rand(2, 4).astype(np.float32)
    x2 = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": np.inf,
        "compute_mode": "use_mm_for_euclid_dist_if_necessary"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cdist"] = cdist_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cdist'.")

check_valid('torch.cdist', generated_inputs['torch.cdist'], lib="torch")

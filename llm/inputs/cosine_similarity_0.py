
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def cosine_similarity_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors, dim=1
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Negative values, dim=1
    x1 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    x2 = np.array([[7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], dtype=np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Different shapes, dim=0
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    dim = 0
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: 3D tensors, dim=2
    x1 = np.random.rand(2, 3, 4).astype(np.float32)
    x2 = np.random.rand(2, 3, 4).astype(np.float32)
    dim = 2
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Integer tensors, dim=1
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.cosine_similarity"] = cosine_similarity_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.cosine_similarity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cosine_similarity'.")

check_valid('torch.nn.functional.cosine_similarity', generated_inputs['torch.nn.functional.cosine_similarity'], lib="torch")


from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, dim=1
    a = torch.randn(4, 3).numpy()
    b = torch.randn(4, 3).numpy()
    dim = 1
    input_dict = {"input": a, "other": b, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors, no dim specified
    a = torch.randn(4, 3).numpy()
    b = torch.randn(4, 3).numpy()
    input_dict = {"input": a, "other": b, "dim": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Float tensors, dim=0
    a = torch.randn(3, 4).numpy()
    b = torch.randn(3, 4).numpy()
    dim = 0
    input_dict = {"input": a, "other": b, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Double tensors, dim=1, providing out
    a = torch.randn(2, 3, dtype=torch.double).numpy()
    b = torch.randn(2, 3, dtype=torch.double).numpy()
    out = torch.zeros(2, 3, dtype=torch.double).numpy()
    dim = 1
    input_dict = {"input": a, "other": b, "dim": dim, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Complex float tensors, dim=0
    a = torch.randn(3, dtype=torch.cfloat).numpy()
    b = torch.randn(3, dtype=torch.cfloat).numpy()
    dim = 0
    input_dict = {"input": a, "other": b, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Complex double tensors, dim=0
    a = torch.randn(3, dtype=torch.cdouble).numpy()
    b = torch.randn(3, dtype=torch.cdouble).numpy()
    dim = 0
    input_dict = {"input": a, "other": b, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Float tensors, 3D, dim=2
    a = torch.randn(2, 2, 3).numpy()
    b = torch.randn(2, 2, 3).numpy()
    dim = 2
    input_dict = {"input": a, "other": b, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: Negative values, float tensors, dim = 1
    a = torch.randn(4, 3) * -1.0
    b = torch.randn(4, 3) * -1.0
    a = a.numpy()
    b = b.numpy()
    dim = 1
    input_dict = {"input": a, "other": b, "dim": dim, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cross"] = cross_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cross'.")

check_valid('torch.cross', generated_inputs['torch.cross'], lib="torch")

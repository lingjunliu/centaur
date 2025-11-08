
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_inputs():
    list_of_inputs = []

    M = torch.randn(3, 3, dtype=torch.float32)
    A = M @ M.mT + 1e-3 * torch.eye(3, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.randn(3, 3, dtype=torch.float64)
    A = M @ M.mT + 1e-6 * torch.eye(3, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": True, "out": out}))

    A = torch.tensor([[2.5]], dtype=torch.float32)
    input_arr = A.numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.randn(2, 2, 2, dtype=torch.float32)
    A = M @ M.mT + 1e-3 * torch.eye(2, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.randn(4, 3, 3, dtype=torch.float64)
    A = M @ M.mT + 1e-6 * torch.eye(3, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": True, "out": out}))

    M = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    A = M @ M.mT + 1e-2 * torch.eye(4, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.randn(5, 5, dtype=torch.float64) * 1e-2
    A = M @ M.mT + 1e-6 * torch.eye(5, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": True, "out": out}))

    M = torch.randn(4, 4, dtype=torch.float32)
    A = M @ M.mT + 1e-4 * torch.eye(4, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.tril(torch.randn(4, 4, dtype=torch.float32))
    A = M @ M.mT + 1e-3 * torch.eye(4, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.randn(3, 5, 5, dtype=torch.float32)
    A = M @ M.mT + 1e-3 * torch.eye(5, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": True, "out": out}))

    d = torch.arange(1, 7, dtype=torch.float64)
    A = torch.diag(d)
    input_arr = A.numpy()
    out = np.empty_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": False, "out": out}))

    M = torch.randn(1, 2, 3, 3, dtype=torch.float64)
    A = M @ M.mT + 1e-5 * torch.eye(3, dtype=M.dtype)
    input_arr = A.numpy()
    out = np.zeros_like(input_arr)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "upper": True, "out": out}))

    return list_of_inputs

generated_inputs["torch.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky'.")


check_valid('torch.cholesky', generated_inputs['torch.cholesky'], lib="torch", suffix=0)

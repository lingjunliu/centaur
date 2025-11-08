
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lu_solve_inputs():
    list_of_inputs = []

    m = 3
    A = torch.randn(m, m, dtype=torch.float32)
    A = A + torch.eye(m, dtype=torch.float32) * 3.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(m, 1, dtype=torch.float32)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    m = 4
    A = torch.randn(m, m, dtype=torch.float64) - 2.0
    A = A + torch.eye(m, dtype=torch.float64) * 5.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(m, 2, dtype=torch.float64) - 1.0
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    batch = 2
    m = 3
    A = torch.randn(batch, m, m, dtype=torch.float32)
    A = A + torch.eye(m, dtype=torch.float32).expand(batch, m, m) * 2.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(batch, m, 1, dtype=torch.float32)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    batch = 3
    m = 5
    A = torch.randn(batch, m, m, dtype=torch.float64)
    A = A + torch.eye(m, dtype=torch.float64).expand(batch, m, m) * 4.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(batch, m, 3, dtype=torch.float64)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    m = 1
    A = torch.tensor([[2.5]], dtype=torch.float32)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.tensor([[3.0]], dtype=torch.float32)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    m = 2
    A = torch.randn(m, m, dtype=torch.complex64)
    A = A + torch.eye(m, dtype=torch.complex64) * (2.0 + 0.5j)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(m, 1, dtype=torch.complex64)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    batch = 2
    m = 3
    A = torch.randn(batch, m, m, dtype=torch.complex128)
    A = A + torch.eye(m, dtype=torch.complex128).expand(batch, m, m) * (3.0 + 1.0j)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(batch, m, 2, dtype=torch.complex128)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    m = 3
    A = torch.randn(m, m, dtype=torch.float32)
    A = A + torch.eye(m, dtype=torch.float32) * 2.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(4, m, 2, dtype=torch.float32)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    batch = (2, 2)
    m = 4
    A = torch.randn(*batch, m, m, dtype=torch.float32)
    A = A + torch.eye(m, dtype=torch.float32).expand(*batch, m, m) * 2.5
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.randn(*batch, m, 3, dtype=torch.float32)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    m = 5
    A = torch.randn(m, m, dtype=torch.float64)
    A = A + torch.eye(m, dtype=torch.float64) * 6.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = torch.zeros(m, 4, dtype=torch.float64)
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    m = 6
    A = -torch.randn(m, m, dtype=torch.float32) * 5.0 - 10.0
    A = A + torch.eye(m, dtype=torch.float32) * 30.0
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b = -torch.randn(m, 1, dtype=torch.float32) * 2.0
    input_dict = {
        "b": b.numpy(),
        "LU_data": LU_data.numpy(),
        "LU_pivots": LU_pivots.to(torch.int32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.lu_solve"] = lu_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_solve'.")


check_valid('torch.lu_solve', generated_inputs['torch.lu_solve'], lib="torch", suffix=0)

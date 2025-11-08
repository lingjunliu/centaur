
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def linalg_solve_inputs():
    list_of_inputs = []

    # Input 1: float32, single system, vector RHS
    n = 3
    A = torch.randn(n, n, dtype=torch.float32)
    A = A + 3.0 * torch.eye(n, dtype=torch.float32)
    B = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32)
    out = torch.zeros(n, dtype=torch.float32)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, single system, multiple RHS
    n, k = 4, 2
    A = torch.randn(n, n, dtype=torch.float64)
    A = A + 4.0 * torch.eye(n, dtype=torch.float64)
    B = torch.randn(n, k, dtype=torch.float64) * 2 - 1.0
    out = torch.zeros(n, k, dtype=torch.float64)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, batched systems, multiple RHS
    bsz, n, k = 2, 3, 4
    A = torch.randn(bsz, n, n, dtype=torch.float32)
    A = A + 3.0 * torch.eye(n, dtype=torch.float32)
    B = torch.randn(bsz, n, k, dtype=torch.float32)
    out = torch.zeros(bsz, n, k, dtype=torch.float32)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, batched systems, broadcasting B vector
    bsz, n = 3, 3
    A = torch.randn(bsz, n, n, dtype=torch.float64)
    A = A + 3.0 * torch.eye(n, dtype=torch.float64)
    B = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    out = torch.zeros(bsz, n, dtype=torch.float64)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, batched systems, broadcasting B column
    bsz, n, k = 2, 5, 1
    A = torch.randn(bsz, n, n, dtype=torch.float32)
    A = A + 5.0 * torch.eye(n, dtype=torch.float32)
    B = torch.randn(n, k, dtype=torch.float32)
    out = torch.zeros(bsz, n, k, dtype=torch.float32)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, left=False, single system
    k = 3
    A = torch.randn(k, k, dtype=torch.float64)
    A = A + 3.0 * torch.eye(k, dtype=torch.float64)
    B = torch.tensor([[1.0, -2.0, 3.0],
                      [-1.5, 0.0, 2.5]], dtype=torch.float64)
    out = torch.zeros(B.shape, dtype=torch.float64)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": False, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64, left=False, multiple RHS
    k, n = 4, 5
    Ar = torch.randn(k, k, dtype=torch.float32)
    Ai = torch.randn(k, k, dtype=torch.float32)
    A = torch.complex(Ar, Ai)
    I = torch.eye(k, dtype=torch.float32)
    A = A + 4.0 * torch.complex(I, torch.zeros_like(I))
    Br = torch.randn(n, k, dtype=torch.float32)
    Bi = torch.randn(n, k, dtype=torch.float32)
    B = torch.complex(Br, Bi)
    out = torch.zeros(n, k, dtype=torch.cfloat)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": False, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, batched systems, multiple RHS (larger batch)
    bsz, n, k = 4, 2, 3
    A = torch.randn(bsz, n, n, dtype=torch.float64)
    A = A + 2.0 * torch.eye(n, dtype=torch.float64)
    B = torch.randn(bsz, n, k, dtype=torch.float64)
    out = torch.zeros(bsz, n, k, dtype=torch.float64)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128, single system, vector RHS
    n = 2
    Ar = torch.randn(n, n, dtype=torch.float64)
    Ai = torch.randn(n, n, dtype=torch.float64)
    A = torch.complex(Ar, Ai)
    I = torch.eye(n, dtype=torch.float64)
    A = A + 2.0 * torch.complex(I, torch.zeros_like(I))
    Br = torch.tensor([1.0, -0.5], dtype=torch.float64)
    Bi = torch.tensor([0.25, 1.5], dtype=torch.float64)
    B = torch.complex(Br, Bi)
    out = torch.zeros(n, dtype=torch.cdouble)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, single system, column RHS
    n, k = 3, 1
    A = torch.randn(n, n, dtype=torch.float32)
    A = A + 3.0 * torch.eye(n, dtype=torch.float32)
    B = torch.randn(n, k, dtype=torch.float32)
    out = torch.zeros(n, k, dtype=torch.float32)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float64, multi-batch dims, vector RHS
    b1, b2, n = 2, 2, 3
    A = torch.randn(b1, b2, n, n, dtype=torch.float64)
    A = A + 3.0 * torch.eye(n, dtype=torch.float64)
    B = torch.randn(b1, b2, n, dtype=torch.float64)
    out = torch.zeros(b1, b2, n, dtype=torch.float64)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float64, batched A with broadcasting B matrix
    bsz, n, k = 4, 2, 2
    A = torch.randn(bsz, n, n, dtype=torch.float64)
    A = A + 2.0 * torch.eye(n, dtype=torch.float64)
    B = torch.randn(n, k, dtype=torch.float64)
    out = torch.zeros(bsz, n, k, dtype=torch.float64)
    input_dict = {"A": A.numpy(), "B": B.numpy(), "left": True, "out": out.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.solve"] = linalg_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve'.")


check_valid('torch.linalg.solve', generated_inputs['torch.linalg.solve'], lib="torch", suffix=0)

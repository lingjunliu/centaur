
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def linalg_inv_inputs():
    list_of_inputs = []
    
    A = torch.eye(2, dtype=torch.float32)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    A = torch.tensor([[3.0]], dtype=torch.float64)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    M = torch.randn(3, 3, dtype=torch.float32)
    A = M @ M.transpose(-1, -2) + 0.5 * torch.eye(3, dtype=torch.float32)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    M = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    A = M @ M.transpose(-1, -2) + 0.1 * torch.eye(4, dtype=torch.float64)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    Mr = torch.randn(3, 3, dtype=torch.float32)
    Mi = torch.randn(3, 3, dtype=torch.float32)
    M = torch.complex(Mr, Mi)
    A = M @ M.conj().transpose(-1, -2) + 0.2 * torch.eye(3, dtype=torch.complex64)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    Mr = torch.randn(2, 2, 2, 2, dtype=torch.float64)
    Mi = torch.randn(2, 2, 2, 2, dtype=torch.float64)
    M = torch.complex(Mr, Mi)
    A = M @ M.conj().transpose(-1, -2) + 1e-3 * torch.eye(2, dtype=torch.complex128)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    vals = torch.tensor([-2.0, -1.5, -0.5], dtype=torch.float32)
    A = torch.diag(vals)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    L = torch.tril(torch.randn(4, 4, dtype=torch.float64))
    A = L + torch.eye(4, dtype=torch.float64)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    I = torch.eye(3, dtype=torch.float32)
    A = torch.stack([I for _ in range(10)], dim=0)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    R = torch.randn(6, 6, dtype=torch.float64)
    Q, _ = torch.linalg.qr(R)
    A = Q
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    A = torch.ones((1, 1, 1), dtype=torch.complex64) * (2.0 + 0.0j)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    M = torch.randn(3, 5, 5, dtype=torch.float64)
    A = M @ M.transpose(-1, -2) + 0.01 * torch.eye(5, dtype=torch.float64)
    out = torch.empty_like(A)
    list_of_inputs.append(copy.deepcopy({"A": A.numpy(), "out": out.numpy()}))
    
    return list_of_inputs

generated_inputs["torch.linalg.inv"] = linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.inv'.")


check_valid('torch.linalg.inv', generated_inputs['torch.linalg.inv'], lib="torch", suffix=0)

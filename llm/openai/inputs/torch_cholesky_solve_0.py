
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def cholesky_solve_inputs():
    list_of_inputs = []

    def _is_complex_dtype(dt):
        return dt in (
            getattr(torch, 'complex64', None),
            getattr(torch, 'complex128', None),
            getattr(torch, 'cfloat', None),
            getattr(torch, 'cdouble', None),
        )

    def _cholesky(A):
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "cholesky"):
            return torch.linalg.cholesky(A)
        return torch.cholesky(A, upper=False)

    def to_numpy(t):
        if hasattr(t, "resolve_conj"):
            t = t.resolve_conj()
        if hasattr(t, "resolve_neg"):
            t = t.resolve_neg()
        return t.detach().cpu().numpy()

    def make_factor(n, batch_shape=(), dtype=torch.float32, upper=False):
        if _is_complex_dtype(dtype):
            Mr = torch.randn(*batch_shape, n, n, dtype=torch.float32)
            Mi = torch.randn(*batch_shape, n, n, dtype=torch.float32)
            M = (Mr + 1j * Mi).to(dtype)
            A = M @ M.conj().transpose(-2, -1) + (n * 1e-1) * torch.eye(n, dtype=dtype).expand(*batch_shape, n, n)
        else:
            M = torch.randn(*batch_shape, n, n, dtype=dtype)
            A = M @ M.transpose(-2, -1) + (n * 1e-1) * torch.eye(n, dtype=dtype).expand(*batch_shape, n, n)
        L_lower = _cholesky(A)
        return L_lower.conj().transpose(-2, -1) if upper else L_lower

    # Input 1
    n, k = 3, 1
    dtype = torch.float32
    L = make_factor(n, dtype=dtype, upper=False)
    B = torch.randn(n, k, dtype=dtype)
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 2
    n, k = 4, 2
    dtype = torch.float64
    L = make_factor(n, dtype=dtype, upper=True)
    B = torch.randn(n, k, dtype=dtype)
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": True, "out": to_numpy(out)}))

    # Input 3
    n, k = 3, 1
    batch = (2,)
    dtype = torch.float32
    L = make_factor(n, batch_shape=batch, dtype=dtype, upper=False)
    B = torch.randn(*batch, n, k, dtype=dtype)
    out = torch.empty(*batch, n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 4
    n, k = 5, 2
    batch_L = (3,)
    batch_B = (1,)
    dtype = torch.float64
    L = make_factor(n, batch_shape=batch_L, dtype=dtype, upper=False)
    B = torch.randn(*batch_B, n, k, dtype=dtype)
    out = torch.empty(*batch_L, n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 5
    n, k = 5, 1
    batch = (2, 2)
    dtype = torch.float32
    L = make_factor(n, batch_shape=batch, dtype=dtype, upper=False)
    B = torch.randn(*batch, n, k, dtype=dtype)
    out = torch.empty(*batch, n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 6
    n, k = 3, 2
    dtype = torch.complex64 if hasattr(torch, "complex64") else torch.cfloat
    L = make_factor(n, dtype=dtype, upper=False)
    Br = torch.randn(n, k, dtype=torch.float32)
    Bi = torch.randn(n, k, dtype=torch.float32)
    B = (Br + 1j * Bi).to(dtype)
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 7
    n, k = 2, 1
    dtype = torch.complex128 if hasattr(torch, "complex128") else torch.cdouble
    L = make_factor(n, dtype=dtype, upper=True)
    Br = torch.randn(n, k, dtype=torch.float64)
    Bi = torch.randn(n, k, dtype=torch.float64)
    B = (Br + 1j * Bi).to(dtype)
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": True, "out": to_numpy(out)}))

    # Input 8
    n, k = 4, 3
    dtype = torch.float64
    L_lower = make_factor(n, dtype=dtype, upper=False)
    U = L_lower.transpose(-2, -1)
    B = torch.randn(n, k, dtype=dtype)
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(U), "upper": True, "out": to_numpy(out)}))

    # Input 9
    n, k = 6, 3
    dtype = torch.float64
    L = make_factor(n, dtype=dtype, upper=False)
    B = torch.randn(n, k, dtype=dtype) * 2 - 1
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 10
    n, k = 1, 1
    dtype = torch.float32
    L = make_factor(n, dtype=dtype, upper=False)
    B = torch.tensor([[-3.5]], dtype=dtype)
    out = torch.empty(n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    # Input 11
    n, k = 3, 4
    dtype = torch.float32
    L = make_factor(n, batch_shape=(1, 5), dtype=dtype, upper=False)
    B = torch.randn(2, 5, n, k, dtype=dtype)
    out = torch.empty(2, 5, n, k, dtype=dtype)
    list_of_inputs.append(copy.deepcopy({"input": to_numpy(B), "L": to_numpy(L), "upper": False, "out": to_numpy(out)}))

    return list_of_inputs

generated_inputs["torch.cholesky_solve"] = cholesky_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cholesky_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky_solve'.")


check_valid('torch.cholesky_solve', generated_inputs['torch.cholesky_solve'], lib="torch", suffix=0)

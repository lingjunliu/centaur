
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def eigvals_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32
    A = torch.tensor([[3.0, 1.0],
                      [0.0, -2.0]], dtype=torch.float32).numpy()
    out = torch.zeros((2,), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 2: 3x3 float64 with negatives
    A = torch.tensor([[0.0, -1.0, 2.0],
                      [4.0, 5.5, -6.1],
                      [-7.2, 8.3, 9.4]], dtype=torch.float64).numpy()
    out = torch.zeros((3,), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 3: 1x1 float32
    A = torch.tensor([[-5.0]], dtype=torch.float32).numpy()
    out = torch.zeros((1,), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 4: batch (2, 2, 2) float32
    A = torch.randn(2, 2, 2, dtype=torch.float32).numpy()
    out = torch.zeros((2, 2), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 5: batch (4, 3, 3) float64
    A = torch.randn(4, 3, 3, dtype=torch.float64).numpy()
    out = torch.zeros((4, 3), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 6: 2x2 complex64
    A = torch.tensor([[1+2j, -1j],
                      [3+0j, 4+5j]], dtype=torch.complex64).numpy()
    out = torch.zeros((2,), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 7: 3x3 complex128 Hermitian
    real = torch.randn(3, 3, dtype=torch.float64)
    imag = torch.randn(3, 3, dtype=torch.float64)
    X = torch.complex(real, imag)
    H = X + X.conj().transpose(-2, -1)
    A = H.numpy()
    out = torch.zeros((3,), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 8: 5x5 float64 random
    A = torch.randn(5, 5, dtype=torch.float64).numpy()
    out = torch.zeros((5,), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 9: batch (2, 3, 4, 4) float32
    A = torch.randn(2, 3, 4, 4, dtype=torch.float32).numpy()
    out = torch.zeros((2, 3, 4), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 10: 3x3 zero matrix float32
    A = torch.zeros(3, 3, dtype=torch.float32).numpy()
    out = torch.zeros((3,), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 11: 3x3 singular with repeated eigenvalues float64
    A = torch.tensor([[2.0, 1.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 2.0]], dtype=torch.float64).numpy()
    out = torch.zeros((3,), dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    # Input 12: 4x4 complex64 diagonal
    A = torch.diag(torch.tensor([1-1j, -2+3j, 0+0j, 4-5j], dtype=torch.complex64)).numpy()
    out = torch.zeros((4,), dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"A": A, "out": out}))

    return list_of_inputs

generated_inputs["torch.linalg.eigvals"] = eigvals_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.eigvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eigvals'.")


check_valid('torch.linalg.eigvals', generated_inputs['torch.linalg.eigvals'], lib="torch", suffix=0)

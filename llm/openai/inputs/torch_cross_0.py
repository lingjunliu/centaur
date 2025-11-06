
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def cross_inputs():
    list_of_inputs = []

    # 1
    input = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).numpy()
    dim = 0
    out = torch.zeros(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 2
    input = torch.randn(4, 3, dtype=torch.float64).numpy()
    other = torch.randn(4, 3, dtype=torch.float64).numpy()
    dim = 1
    out = torch.zeros(4, 3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 3
    input = torch.randn(3, 4, dtype=torch.float32).numpy()
    other = torch.randn(3, 4, dtype=torch.float32).numpy()
    dim = 0
    out = torch.zeros(3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 4
    input = torch.randn(2, 3, 5, dtype=torch.float64).numpy()
    other = torch.randn(2, 3, 5, dtype=torch.float64).numpy()
    dim = 1
    out = torch.zeros(2, 3, 5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 5
    input = torch.randn(2, 5, 3, dtype=torch.float32).numpy()
    other = torch.randn(2, 5, 3, dtype=torch.float32).numpy()
    dim = -1
    out = torch.zeros(2, 5, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 6 (complex64)
    a_real = torch.randn(2, 4, 3, 6, dtype=torch.float32)
    a_imag = torch.randn(2, 4, 3, 6, dtype=torch.float32)
    b_real = torch.randn(2, 4, 3, 6, dtype=torch.float32)
    b_imag = torch.randn(2, 4, 3, 6, dtype=torch.float32)
    input = (a_real + 1j * a_imag).numpy()
    other = (b_real + 1j * b_imag).numpy()
    dim = 2
    out = torch.zeros(2, 4, 3, 6, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 7
    input = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64).numpy()
    other = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64).numpy()
    dim = 2
    out = torch.zeros(1, 2, 3, 4, 5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 8 (complex128)
    a_real = torch.randn(7, 3, dtype=torch.float64)
    a_imag = torch.randn(7, 3, dtype=torch.float64)
    b_real = torch.randn(7, 3, dtype=torch.float64)
    b_imag = torch.randn(7, 3, dtype=torch.float64)
    input = (a_real + 1j * a_imag).numpy()
    other = (b_real + 1j * b_imag).numpy()
    dim = 1
    out = torch.zeros(7, 3, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 9
    input = torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float64).repeat(5, 1).numpy()
    other = torch.tensor([[4.0, -5.0, 6.0]], dtype=torch.float64).repeat(5, 1).numpy()
    dim = 1
    out = torch.zeros(5, 3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 10
    input = torch.randn(6, 3, 2, dtype=torch.float32).numpy()
    other = torch.randn(6, 3, 2, dtype=torch.float32).numpy()
    dim = 1
    out = torch.zeros(6, 3, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 11
    input = torch.randn(3, 3, 3, dtype=torch.float32).numpy()
    other = torch.randn(3, 3, 3, dtype=torch.float32).numpy()
    dim = 2
    out = torch.zeros(3, 3, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    # 12
    input = torch.randn(5, 2, 3, 4, dtype=torch.float64).numpy()
    other = torch.randn(5, 2, 3, 4, dtype=torch.float64).numpy()
    dim = -2
    out = torch.zeros(5, 2, 3, 4, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "dim": dim, "out": out}))

    return list_of_inputs

generated_inputs["torch.cross"] = cross_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cross'.")


check_valid('torch.cross', generated_inputs['torch.cross'], lib="torch", suffix=0)

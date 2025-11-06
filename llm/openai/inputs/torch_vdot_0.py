
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def vdot_inputs():
    list_of_inputs = []

    input = torch.tensor([2, 3, -4], dtype=torch.int64).numpy()
    other = torch.tensor([5, 0, 2], dtype=torch.int64).numpy()
    out = torch.tensor(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-1.0, 2.5, 3.0, -4.5, 0.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.5, -2.0, 4.0, 1.5, 2.0], dtype=torch.float32).numpy()
    out = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64).numpy()
    other = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64).numpy()
    out = torch.tensor(0.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([1+2j, 3-1j], dtype=torch.complex64).numpy()
    other = torch.tensor([2+1j, 4+0j], dtype=torch.complex64).numpy()
    out = torch.tensor(0, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([1-1j, -2.5+0.5j, 3+4j], dtype=torch.complex128).numpy()
    other = torch.tensor([2+0j, -3+0j, 0.5+0j], dtype=torch.complex128).numpy()
    out = torch.tensor(0, dtype=torch.complex128).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([7], dtype=torch.int32).numpy()
    other = torch.tensor([-3], dtype=torch.int32).numpy()
    out = torch.tensor(0, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    base1 = torch.arange(0, 10, dtype=torch.float64)
    base2 = torch.linspace(1.0, 2.0, steps=10, dtype=torch.float64)
    input = base1[1::2].numpy()
    other = base2[1::2].numpy()
    out = torch.tensor(0.0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.randn(1000, dtype=torch.float16).numpy()
    other = torch.randn(1000, dtype=torch.float16).numpy()
    out = torch.tensor(0.0, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-100, 200, 300, -400], dtype=torch.int16).numpy()
    other = torch.tensor([1, -2, 3, -4], dtype=torch.int16).numpy()
    out = torch.tensor(0, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(-64, 64, dtype=torch.int64).numpy()
    other = torch.arange(64, -64, -1, dtype=torch.int64).flip(0).numpy()
    out = torch.tensor(0, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    a = torch.tensor([1+1j, 2+0j, 3-1j, 4+2j, 5+0j, 6-3j], dtype=torch.complex64)
    b = torch.tensor([0.5-0.5j, -1+2j, 3+0j, 4+1j, 5-2j, 6+0j], dtype=torch.complex64)
    input = a[::2].numpy()
    other = b[::2].numpy()
    out = torch.tensor(0, dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([1e10, -1e10, 3.14159, -2.71828], dtype=torch.float32).numpy()
    other = torch.tensor([1e-3, 2e-3, -3.14159e-3, 2.71828e-3], dtype=torch.float32).numpy()
    out = torch.tensor(0.0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.vdot"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.vdot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vdot'.")


check_valid('torch.vdot', generated_inputs['torch.vdot'], lib="torch", suffix=0)

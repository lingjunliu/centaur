
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def conj_physical_inputs():
    list_of_inputs = []

    t = torch.tensor([1+2j, -3-4j, 0+0j], dtype=torch.complex64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[1-2j, -0.0+3j, -4-0.5j],
                      [5+0j, 6-7j, -8+9j]], dtype=torch.complex128)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([0.0, -1.5, 3.2, float('nan'), float('inf'), -float('inf')], dtype=torch.float32)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[1.0, -2.5, 3.75],
                      [4.125, -5.5, 0.0]], dtype=torch.float64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.arange(-12, 12, dtype=torch.int32).reshape(2, 2, 6)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor(1234567890123456789, dtype=torch.int64)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.arange(2*3*4*1, dtype=torch.uint8).reshape(2, 3, 4, 1)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.randn((2, 3, 4), dtype=torch.cfloat)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.empty((0, 5), dtype=torch.complex128)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([[1.5, -2.0, 3.0],
                      [4.0, -5.5, 6.25],
                      [0.0, -0.5, 8.0]], dtype=torch.float16)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    t = torch.tensor(3-4j, dtype=torch.complex128)
    input = t.numpy()
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.conj_physical"] = conj_physical_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.conj_physical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.conj_physical'.")


check_valid('torch.conj_physical', generated_inputs['torch.conj_physical'], lib="torch", suffix=0)

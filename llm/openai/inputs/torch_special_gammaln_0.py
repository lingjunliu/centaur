
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def gammaln_inputs():
    list_of_inputs = []

    ti = torch.tensor([0.5, 1.0, 2.5, 5.0], dtype=torch.float32)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor([-0.5, -1.3, 0.0, 1.0, 3.0], dtype=torch.float64)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor([[0.1, 0.2, 0.3],
                       [4.5, 5.5, 6.5]], dtype=torch.float16)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor([[[ -0.75, 0.25, 1.25, 2.75]],
                       [[ 3.5 , -2.5 , 0.5 , 10.0]]], dtype=torch.float32)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor(3.5, dtype=torch.float64)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.linspace(0.1, 5.0, steps=24, dtype=torch.float32).reshape(2, 2, 2, 3)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    base = (torch.arange(15, dtype=torch.float64).reshape(3, 5) / 10.0) + 0.1
    ti = base[:, ::2]
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor([-0.99999, -1.00001, -1.99999, -2.00001, -10.5, -0.1, 0.1], dtype=torch.float64)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor([20.0, 50.0, 100.0, 170.0], dtype=torch.float64)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.tensor([1e-8, 1e-6, 1e-4, 1e-2], dtype=torch.float64)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.empty((0,), dtype=torch.float32)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    ti = torch.empty((2, 0), dtype=torch.float32)
    input = ti.numpy()
    out = torch.empty_like(ti).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))

    return list_of_inputs

generated_inputs["torch.special.gammaln"] = gammaln_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.gammaln' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.gammaln'.")


check_valid('torch.special.gammaln', generated_inputs['torch.special.gammaln'], lib="torch", suffix=0)

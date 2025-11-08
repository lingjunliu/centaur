
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def deg2rad_inputs():
    list_of_inputs = []
    default_float = torch.get_default_dtype()

    t = torch.tensor([0.0, 90.0, 180.0, -90.0], dtype=torch.float32)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]], dtype=torch.float64)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor(45.0, dtype=torch.float32)
    out = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor([[[0, 30, 60], [90, 120, 150]]], dtype=torch.int64)
    out = torch.empty_like(t, dtype=default_float).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor([[-720.0, -360.0, 0.0, 360.0, 720.0]], dtype=torch.float16)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.linspace(-540, 540, steps=24, dtype=torch.float32).reshape(2, 3, 2, 2)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.empty(0, dtype=torch.float32)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor([float('nan'), float('inf'), float('-inf'), 45.0, -270.0], dtype=torch.float32)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor([[1, 2, 3], [-1, -2, -3]], dtype=torch.int32)
    out = torch.empty_like(t, dtype=default_float).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.tensor([0, 90, 180, 200, 255], dtype=torch.uint8)
    out = torch.empty_like(t, dtype=default_float).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    t = torch.linspace(-1000, 1000, steps=100, dtype=torch.float64)
    out = torch.empty_like(t).numpy()
    list_of_inputs.append(copy.deepcopy({"input": t.numpy(), "out": out}))

    return list_of_inputs

generated_inputs["torch.deg2rad"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.deg2rad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.deg2rad'.")


check_valid('torch.deg2rad', generated_inputs['torch.deg2rad'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def mse_loss_inputs():
    list_of_inputs = []

    inp = torch.randn(3, 5, dtype=torch.float32).numpy()
    tgt = torch.randn(3, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }))

    inp = torch.linspace(-5, 5, steps=10, dtype=torch.float64).numpy()
    tgt = torch.linspace(5, -5, steps=10, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": inp,
        "target": tgt
    }))

    inp = torch.tensor(1.2345, dtype=torch.float32).numpy()
    tgt = torch.tensor(-0.9876, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }))

    inp = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    tgt = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }))

    inp = torch.randn(1, 2, 3, 4, dtype=torch.float32).numpy()
    tgt = torch.randn(1, 2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": inp,
        "target": tgt
    }))

    inp = torch.zeros(2, 1, 2, 1, 2, dtype=torch.float32).numpy()
    tgt = torch.ones(2, 1, 2, 1, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }))

    inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    tgt = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }))

    inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).numpy()
    tgt = torch.tensor([[4.0, 3.0], [2.0, 1.0]], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }))

    inp = (torch.randn(4, 4, 4, dtype=torch.float32) * 100.0).numpy()
    tgt = (torch.randn(4, 4, 4, dtype=torch.float32) * -50.0).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }))

    inp = torch.randn(3, 1, dtype=torch.float16).numpy()
    tgt = torch.randn(3, 1, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": inp,
        "target": tgt
    }))

    inp = torch.arange(6, dtype=torch.float32).reshape(6).numpy()
    tgt = torch.arange(6, 0, -1, dtype=torch.float32).reshape(6).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": False,
        "reduce": True,
        "reduction": "none",
        "input": inp,
        "target": tgt
    }))

    inp = torch.full((2, 2), 3.14, dtype=torch.float64).numpy()
    tgt = torch.full((2, 2), 3.14, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": inp,
        "target": tgt
    }))

    return list_of_inputs

generated_inputs["torch.nn.MSELoss"] = mse_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MSELoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MSELoss'.")


check_valid('torch.nn.MSELoss', generated_inputs['torch.nn.MSELoss'], lib="torch", suffix=0)

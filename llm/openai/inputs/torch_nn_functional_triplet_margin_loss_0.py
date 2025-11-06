
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def triplet_margin_loss_inputs():
    list_of_inputs = []

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    p = torch.tensor([1.1, 2.0, 3.0], dtype=torch.float32)
    n = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 1.0,
        "p": 2.0,
        "eps": 1e-6,
        "swap": False,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[-1.0, 0.5, 2.0],
                      [3.0, -4.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[-0.8, 0.4, 2.2],
                      [2.9, -3.9, 0.1]], dtype=torch.float32)
    n = torch.tensor([[1.0, -0.5, -2.0],
                      [-3.0, 4.0, 0.0]], dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.5,
        "p": 1.0,
        "eps": 1e-6,
        "swap": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    a = base
    p = base + 0.1
    n = base - 0.2
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.2,
        "p": 2.0,
        "eps": 1e-12,
        "swap": False,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[0.01, -0.01, 0.02, -0.02]], dtype=torch.float32)
    n = torch.tensor([[2.0, -2.0, 3.0, -3.0]], dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.5,
        "p": 3.0,
        "eps": 0.0,
        "swap": False,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.zeros((3, 3), dtype=torch.float32)
    p = torch.full((3, 3), 0.01, dtype=torch.float32)
    n = torch.full((3, 3), 10.0, dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 2.0,
        "p": 2.0,
        "eps": 1e-6,
        "swap": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[0.0, 1.0],
                      [2.0, 3.0],
                      [4.0, 5.0],
                      [6.0, 7.0],
                      [8.0, 9.0]], dtype=torch.float32)
    p = a + 0.5
    n = a - 0.5
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.3,
        "p": 1.0,
        "eps": 1e-9,
        "swap": True,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    torch.manual_seed(0)
    a = torch.randn(10, 3, dtype=torch.float32)
    p = a + 0.05 * torch.randn(10, 3, dtype=torch.float32)
    n = -a + 0.1 * torch.randn(10, 3, dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.7,
        "p": 2.0,
        "eps": 1e-6,
        "swap": True,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[100.0, 200.0, -300.0, 400.0],
                      [-500.0, 600.0, -700.0, 800.0],
                      [900.0, -1000.0, 1100.0, -1200.0],
                      [1300.0, -1400.0, 1500.0, -1600.0],
                      [1700.0, -1800.0, 1900.0, -2000.0],
                      [2100.0, -2200.0, 2300.0, -2400.0]], dtype=torch.float32)
    p = a + torch.tensor([[-1.0, -0.5, 1.0, 1.0]], dtype=torch.float32).expand_as(a)
    n = -a
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 5.0,
        "p": 2.0,
        "eps": 1e-6,
        "swap": False,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[0.3],
                      [-0.7],
                      [1.1]], dtype=torch.float32)
    p = torch.tensor([[0.25],
                      [-0.65],
                      [1.0]], dtype=torch.float32)
    n = torch.tensor([[1.3],
                      [-1.7],
                      [2.1]], dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.1,
        "p": 1.0,
        "eps": 1e-8,
        "swap": False,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.linspace(0, 1, steps=16, dtype=torch.float64).reshape(2, 8)
    p = a + 0.02
    n = a - 0.03
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.5,
        "p": 2.0,
        "eps": 1e-6,
        "swap": True,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (torch.arange(21, dtype=torch.float32).reshape(7, 3) - 10.0) / 3.0
    a = base
    p = base + 0.05
    n = -base
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.9,
        "p": 2.0,
        "eps": 1e-6,
        "swap": False,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.tensor([[0.1, -0.1],
                      [0.2, -0.2],
                      [0.3, -0.3],
                      [0.4, -0.4]], dtype=torch.float32)
    p = torch.tensor([[0.11, -0.09],
                      [0.19, -0.21],
                      [0.31, -0.29],
                      [0.41, -0.39]], dtype=torch.float32)
    n = torch.tensor([[1.0, -1.0],
                      [1.2, -1.2],
                      [1.3, -1.3],
                      [1.4, -1.4]], dtype=torch.float32)
    input_dict = {
        "anchor": a.numpy(),
        "positive": p.numpy(),
        "negative": n.numpy(),
        "margin": 0.01,
        "p": 1.0,
        "eps": 1e-8,
        "swap": True,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.triplet_margin_loss"] = triplet_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.triplet_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.triplet_margin_loss'.")


check_valid('torch.nn.functional.triplet_margin_loss', generated_inputs['torch.nn.functional.triplet_margin_loss'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def hinge_embedding_loss_inputs():
    list_of_inputs = []

    input = torch.tensor([1.5, -0.5, 0.0, 2.0, -3.0], dtype=torch.float32).numpy()
    target = torch.tensor([1, -1, 1, -1, 1], dtype=torch.float32).numpy()
    margin = 1.0
    reduction = "mean"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.tensor([[0.2, -1.0, 3.5],
                          [4.2, 0.0, -0.7]], dtype=torch.float64).numpy()
    target = torch.tensor([[1, -1, 1],
                           [-1, 1, -1]], dtype=torch.float64).numpy()
    margin = 0.0
    reduction = "sum"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = (torch.arange(8, dtype=torch.float32).view(2, 2, 2) - 3.0).numpy()
    target = torch.ones((2, 2, 2), dtype=torch.float32).numpy()
    margin = 2.5
    reduction = "none"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.tensor(0.7, dtype=torch.float64).numpy()
    target = torch.tensor(-1.0, dtype=torch.float64).numpy()
    margin = 1.0
    reduction = "mean"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.tensor([-0.1, -2.0, -5.5, -0.001], dtype=torch.float32).numpy()
    target = torch.tensor([-1, -1, -1, -1], dtype=torch.float32).numpy()
    margin = 10.0
    reduction = "mean"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    torch.manual_seed(0)
    input = torch.randn(1, 2, 3, 4, dtype=torch.float16).numpy()
    tgt_raw = torch.randint(0, 2, (1, 2, 3, 4))
    target = (tgt_raw * 2 - 1).to(dtype=torch.float16).numpy()
    margin = 0.5
    reduction = "sum"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.zeros((3, 3), dtype=torch.float32).numpy()
    target = torch.tensor([[1, -1, 1],
                           [-1, 1, -1],
                           [1, -1, 1]], dtype=torch.float32).numpy()
    margin = 1.0
    reduction = "none"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    input = base.t().numpy()
    cond = base.t() > 5
    target = torch.where(cond, torch.ones_like(base.t()), -torch.ones_like(base.t())).to(torch.float32).numpy()
    margin = 1.2
    reduction = "mean"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.linspace(-1, 1, steps=12, dtype=torch.float32).view(2, 1, 1, 3, 2).numpy()
    alt = (torch.arange(12) % 2) * 2 - 1
    target = alt.to(torch.float32).view(2, 1, 1, 3, 2).numpy()
    margin = 0.1
    reduction = "none"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    torch.manual_seed(1)
    input = torch.randn(10, dtype=torch.float64).numpy()
    target = (torch.randint(0, 2, (10,)) * 2 - 1).to(torch.float64).numpy()
    margin = 3.0
    reduction = "sum"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.tensor([[0.3], [-0.8], [1.2], [0.0], [-2.5]], dtype=torch.float32).numpy()
    target = torch.tensor([[1], [-1], [1], [-1], [1]], dtype=torch.float32).numpy()
    margin = 0.75
    reduction = "mean"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    input = torch.tensor([float("inf"), -float("inf"), float("nan")], dtype=torch.float32).numpy()
    target = torch.tensor([1, -1, 1], dtype=torch.float32).numpy()
    margin = 1.0
    reduction = "none"
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "margin": margin, "reduction": reduction}))

    return list_of_inputs

generated_inputs["torch.nn.functional.hinge_embedding_loss"] = hinge_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hinge_embedding_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hinge_embedding_loss'.")


check_valid('torch.nn.functional.hinge_embedding_loss', generated_inputs['torch.nn.functional.hinge_embedding_loss'], lib="torch", suffix=0)

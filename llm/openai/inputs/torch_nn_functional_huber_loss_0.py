
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def huber_loss_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    target = torch.tensor([0.0, -1.5, 3.0], dtype=torch.float32).numpy()
    delta = 1.0
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.ones((2, 3), dtype=torch.float32).numpy()
    target = torch.zeros((2, 3), dtype=torch.float32).numpy()
    delta = 0.5
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.arange(-6, 6, dtype=torch.float32).reshape(2, 2, 3).numpy()
    target = torch.arange(0, 12, dtype=torch.float32).reshape(2, 2, 3).mul(0.1).numpy()
    delta = 1.5
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1).div(5.0).numpy()
    target = torch.tensor(0.0, dtype=torch.float32).numpy()
    delta = 1.0
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    input = base.T
    target = (base * 0.25).T
    delta = 0.7
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5], dtype=torch.float64).numpy()
    target = torch.tensor([0.5, -1.5, 2.5, -2.5, 0.0], dtype=torch.float64).numpy()
    delta = 2.0
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float16).numpy()
    target = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float16).numpy()
    delta = 0.2
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.tensor([[1e6, -1e6], [5e5, -5e5]], dtype=torch.float32).numpy()
    target = torch.tensor([[9e5, -9e5], [4.5e5, -4.5e5]], dtype=torch.float32).numpy()
    delta = 1000.0
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.arange(-12, 12, dtype=torch.float32).reshape(2, 1, 12).numpy()
    target = torch.tensor([[[-1.0] * 12]], dtype=torch.float32).numpy()
    delta = 0.7
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.linspace(-1.0, 1.0, steps=120, dtype=torch.float32).reshape(2, 3, 4, 5).numpy()
    target = torch.zeros((2, 3, 4, 5), dtype=torch.float32).numpy()
    delta = 0.1
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    input = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5).div(10.0).numpy()
    target = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1).numpy()
    delta = 1.0
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    base2 = torch.arange(40, dtype=torch.float32).reshape(5, 8).numpy()
    input = base2[::2, ::2]
    target = base2[::2, ::2] - 1.0
    delta = 0.9
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({"input": input, "target": target, "delta": delta, "reduction": reduction}))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.huber_loss"] = huber_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.huber_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.huber_loss'.")


check_valid('torch.nn.functional.huber_loss', generated_inputs['torch.nn.functional.huber_loss'], lib="torch", suffix=0)

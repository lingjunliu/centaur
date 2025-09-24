
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def xlog1py_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32).numpy()
    other = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float32).numpy()
    out = torch.empty(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[1.0, -1.5, 0.0],
                          [3.2, -0.7, 5.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[-0.9999, 0.0, 10.0],
                          [-0.5, 0.2, 3.5]], dtype=torch.float64).numpy()
    out = torch.empty((2, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64).numpy()
    other = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64).numpy()
    out = torch.empty((3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor(2.5, dtype=torch.float64).numpy()
    other = torch.tensor(-0.3, dtype=torch.float64).numpy()
    out = torch.empty((), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[[0.5, 1.0, 2.0],
                           [3.0, 4.0, 5.0]],
                          [[6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0]]], dtype=torch.float16).numpy()
    other = torch.tensor([[[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]],
                          [[0.7, 0.8, 0.9],
                           [1.0, 1.1, 1.2]]], dtype=torch.float16).numpy()
    out = torch.empty((2, 2, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    t = torch.arange(6.0, dtype=torch.float32).reshape(3, 2)
    input = t.t().numpy()
    other = (t.t() + 1.0).numpy()
    out = torch.empty((2, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([1e20, -1e20], dtype=torch.float64).numpy()
    other = torch.tensor([1e-12, 1e-12], dtype=torch.float64).numpy()
    out = torch.empty(2, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([1e-12, -1e-12, 3.0], dtype=torch.float64).numpy()
    other = torch.tensor([-1.0 + 1e-8, -0.9999999, -0.999], dtype=torch.float64).numpy()
    out = torch.empty(3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    base = torch.linspace(-2.0, 2.0, steps=10, dtype=torch.float32)
    input = base[::2].numpy()
    other = torch.linspace(0.1, 0.5, steps=5, dtype=torch.float32).numpy()
    out = torch.empty(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([0.5, -1.0, 2.0, -3.5], dtype=torch.float32).numpy()
    other = torch.tensor([[[0.1, 0.2, 0.3, 0.4]],
                          [[-0.5, -0.4, -0.3, -0.2]]], dtype=torch.float32).numpy()
    out = torch.empty((2, 1, 4), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.empty((0,), dtype=torch.float32).numpy()
    other = torch.empty((0,), dtype=torch.float32).numpy()
    out = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float64).numpy()
    other = torch.tensor([0.5, -0.2, 10.0], dtype=torch.float64).numpy()
    out = torch.empty(3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py_1"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlog1py_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlog1py_1'.")


check_valid('torch.special.xlog1py', generated_inputs['torch.special.xlog1py_1'], lib="torch", suffix=1)

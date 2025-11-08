
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def erfc_inputs():
    list_of_inputs = []
    
    input = torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.0], dtype=torch.float32).numpy()
    out = torch.zeros_like(torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.0], dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[-10.0, -1.0, 0.0, 1.0, 10.0],
                          [1e-6, 1e-3, 1.0, 3.5, 20.0]], dtype=torch.float64).numpy()
    out = torch.zeros((2, 5), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor(-1.2345, dtype=torch.float32).numpy()
    out = torch.zeros((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[[-0.1, 0.0, 0.1],
                           [1.5, -2.0, 3.2]],
                          [[-5.0, 5.0, 10.0],
                           [20.0, -20.0, 0.3]]], dtype=torch.float16).numpy()
    out = torch.zeros((2, 2, 3), dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    a = torch.arange(12.0, dtype=torch.float64).view(3, 4).t().contiguous()
    input = a.numpy()
    out = torch.zeros_like(a).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    out = torch.empty((0, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([-float("inf"), -10.0, 0.0, 10.0, float("inf"), float("nan")], dtype=torch.float32).numpy()
    out = torch.zeros(6, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = (torch.arange(6, dtype=torch.float32).view(1, 2, 1, 3) - 2.5).numpy()
    out = torch.zeros((1, 2, 1, 3), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([1e2, 1e3, 1e4], dtype=torch.float64).numpy()
    out = torch.zeros(3, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = (torch.arange(4, dtype=torch.float32).view(2, 1, 1, 2, 1) - 1.5).numpy()
    out = torch.zeros((2, 1, 1, 2, 1), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([-0.0], dtype=torch.float64).numpy()
    out = torch.zeros(1, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    input = torch.tensor([[1e-16, -1e-16, 5e-17]], dtype=torch.float64).numpy()
    out = torch.zeros((1, 3), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.erfc"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.erfc'.")


check_valid('torch.special.erfc', generated_inputs['torch.special.erfc'], lib="torch", suffix=0)

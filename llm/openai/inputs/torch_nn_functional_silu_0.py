
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def silu_inputs():
    list_of_inputs = []
    
    input_arr = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.tensor([[0.5, -0.5, 2.0], [-2.5, 3.0, -3.5]], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.linspace(-2, 2, steps=8, dtype=torch.float32).reshape(2, 1, 4).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.tensor(2.5, dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.tensor([100.0, -100.0, 20.0, -20.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = (torch.rand((1, 3, 2, 2), dtype=torch.float16) * 4 - 2).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0], dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.randn((2, 1, 1, 2, 3), dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.zeros((4, 0, 5), dtype=torch.float32).numpy()
    input_dict = {"input": input_arr, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = torch.tensor([1e-6, -1e-6, 1e-12, -1e-12], dtype=torch.float64).numpy()
    input_dict = {"input": input_arr, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.silu"] = silu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.silu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.silu'.")


check_valid('torch.nn.functional.silu', generated_inputs['torch.nn.functional.silu'], lib="torch", suffix=0)

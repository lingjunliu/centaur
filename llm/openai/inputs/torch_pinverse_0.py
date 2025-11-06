
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def pinverse_inputs():
    list_of_inputs = []
    
    input = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=torch.float32).numpy()
    rcond = 1e-15
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[-1.0, 2.0],
                          [-3.0, 4.0],
                          [5.0, -6.0]], dtype=torch.float64).numpy()
    rcond = 1e-12
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=torch.float64).numpy()
    rcond = 1e-6
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[1.0, 2.0, 3.0],
                          [2.0, 4.0, 6.0],
                          [3.0, 6.0, 9.0]], dtype=torch.float32).numpy()
    rcond = 1e-15
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]],
                          [[1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0]]], dtype=torch.float32).numpy()
    rcond = 1e-12
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.randn(4, 2, 3, dtype=torch.float32).numpy()
    rcond = 1e-3
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    a = torch.tensor([[1.0, -2.0],
                      [0.5, 3.0]], dtype=torch.float32)
    b = torch.tensor([[0.1, 0.2],
                      [-0.3, 0.4]], dtype=torch.float32)
    input = (a + 1j * b).numpy()
    rcond = 1e-7
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[1.0, -1.0, 2.0],
                          [0.0, 3.0, -3.0],
                          [4.0, 5.0, 6.0],
                          [-2.0, 0.5, 1.5],
                          [7.0, -8.0, 9.0]], dtype=torch.float64).numpy()
    rcond = 0.0
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0],
         [7.0, 8.0]],
        [[-1.0, 0.0],
         [0.0, -1.0],
         [1.0, 1.0],
         [2.0, -2.0]],
        [[0.1, -0.2],
         [0.3, -0.4],
         [0.5, -0.6],
         [0.7, -0.8]]
    ], dtype=torch.float32).numpy()
    rcond = 1e-3
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.randn(2, 2, 3, 3, dtype=torch.float32).numpy()
    rcond = 1e-8
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    base = torch.tensor([[1.0, 2.0, 3.0],
                         [0.0, 1.0, 4.0],
                         [5.0, 6.0, 0.0]], dtype=torch.float64)
    input = (base * 1e-8).numpy()
    rcond = 1e-4
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32).numpy()
    rcond = 0.5
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    input = torch.tensor([[1.0],
                          [0.0],
                          [-1.0],
                          [2.0]], dtype=torch.float32).numpy()
    rcond = 1e-2
    list_of_inputs.append(copy.deepcopy({"input": input, "rcond": rcond}))
    
    return list_of_inputs

generated_inputs["torch.pinverse"] = pinverse_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pinverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pinverse'.")


check_valid('torch.pinverse', generated_inputs['torch.pinverse'], lib="torch", suffix=0)

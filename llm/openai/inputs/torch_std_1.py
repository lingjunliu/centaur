
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def std_inputs():
    list_of_inputs = []
    
    # Input 1
    input_arr = torch.tensor([1.0, 2.0, 3.0, 4.0, -5.0], dtype=torch.float32).numpy()
    dim = 0
    correction = 0
    keepdim = False
    out = torch.empty((), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 2
    input_arr = torch.tensor([[0.5, -1.2, 3.3], [4.4, 5.5, -6.6]], dtype=torch.float64).numpy()
    dim = -1
    correction = 1
    keepdim = True
    out = torch.empty((2, 1), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 3
    input_arr = torch.randn(3, 4, 5, dtype=torch.float32).numpy()
    dim = 1
    correction = 1
    keepdim = False
    out = torch.empty((3, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 4
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    dim = 2
    correction = 0
    keepdim = True
    out = torch.empty((2, 3, 1, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 5
    input_arr = torch.tensor([[1.0], [2.5], [-3.5], [4.0]], dtype=torch.float64).numpy()
    dim = 1
    correction = 1
    keepdim = False
    out = torch.empty((4,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 6
    input_arr = torch.randn(6, 7, 8, dtype=torch.float32).numpy()
    dim = -2
    correction = 0
    keepdim = True
    out = torch.empty((6, 1, 8), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 7
    input_arr = torch.randn(1, 2, 3, 4, 5, dtype=torch.float32).numpy()
    dim = 3
    correction = 2
    keepdim = False
    out = torch.empty((1, 2, 3, 5), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 8
    input_arr = torch.tensor([10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 70.0, -80.0, 90.0, -100.0], dtype=torch.float64).numpy()
    dim = -1
    correction = 0
    keepdim = True
    out = torch.empty((1,), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 9
    input_arr = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    dim = -3
    correction = 0
    keepdim = True
    out = torch.empty((1, 3, 4), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    # Input 10
    input_arr = torch.ones((2, 2, 2, 2), dtype=torch.float32).numpy()
    dim = 0
    correction = 0
    keepdim = True
    out = torch.empty((1, 2, 2, 2), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dim": dim, "correction": correction, "keepdim": keepdim, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.std_1"] = std_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.std_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_1'.")


check_valid('torch.std', generated_inputs['torch.std_1'], lib="torch", suffix=1)

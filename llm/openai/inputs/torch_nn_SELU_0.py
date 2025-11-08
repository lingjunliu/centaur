
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def selu_inputs():
    list_of_inputs = []
    
    # Input 1: 1D, mixed values, float32
    inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    # Input 2: 2D, random normal, float32
    inp = torch.randn(2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": inp}))
    
    # Input 3: 3D, uniform range including negatives, float32
    inp = (torch.rand(1, 3, 4, dtype=torch.float32) * 10 - 5).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    # Input 4: 0D scalar, float64
    inp = torch.tensor(2.5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": inp}))
    
    # Input 5: 4D tensor, float64
    inp = torch.randn(2, 3, 4, 5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    # Input 6: Empty 1D tensor, float32
    inp = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    # Input 7: 1D with extreme values, float32
    inp = torch.tensor([-100.0, -5.0, 0.0, 5.0, 100.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": inp}))
    
    # Input 8: 2D zeros, float32
    inp = torch.zeros(3, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    # Input 9: 5D tensor, float32
    inp = torch.randn(1, 2, 1, 2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": inp}))
    
    # Input 10: 1D with NaN and Inf, float32
    inp = torch.tensor([float('nan'), float('inf'), -float('inf'), -1.2345], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    # Input 11: 2D transposed view, float32
    inp = torch.randn(4, 3, dtype=torch.float32).t().numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": True, "input": inp}))
    
    # Input 12: 3D, small values around zero, float64
    inp = (torch.randn(2, 2, 2, dtype=torch.float64) * 1e-3).numpy()
    list_of_inputs.append(copy.deepcopy({"inplace": False, "input": inp}))
    
    return list_of_inputs

generated_inputs["torch.nn.SELU"] = selu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.SELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SELU'.")


check_valid('torch.nn.SELU', generated_inputs['torch.nn.SELU'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lu_solve_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [3.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0, 0.0], [4.0, 1.0, 0.0], [7.0, 8.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    b = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [3.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    b = torch.tensor([[0.5, 1.5], [2.5, 3.5]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [2.5, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0, 0.0], [4.0, 1.0, 0.0], [7.0, 8.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    b = torch.tensor([[2.0, 3.0], [4.0, 5.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [4.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0, 0.0], [4.0, 1.0, 0.0], [7.0, 8.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    b = torch.tensor([[2.0, 3.0], [4.0, 5.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [4.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [3.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU = torch.tensor([[1.0, 0.0], [3.0, 1.0]]).numpy()
    
    input_dict = {
        "b": b,
        "LU": LU
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lu_solve"] = lu_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_solve'.")


check_valid('torch.lu_solve', generated_inputs['torch.lu_solve'], lib="torch", suffix=0)

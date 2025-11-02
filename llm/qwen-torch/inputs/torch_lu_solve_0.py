
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lu_solve_inputs():
    list_of_inputs = []
    
    # Input 1
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
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

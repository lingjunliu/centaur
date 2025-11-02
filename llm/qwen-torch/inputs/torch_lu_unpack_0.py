
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lu_unpack_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    LU_data = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    unpack_data = True
    unpack_pivots = True
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    LU_data = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([0, 1], dtype=torch.int32).numpy()
    unpack_data = False
    unpack_pivots = True
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    LU_data = torch.tensor([[-1.0, -2.0, -3.0],
                           [4.0, 5.0, 6.0]]).numpy()
    LU_pivots = torch.tensor([3, 2], dtype=torch.int32).numpy()
    unpack_data = True
    unpack_pivots = False
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    LU_data = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0],
                           [10.0, 11.0, 12.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    unpack_data = False
    unpack_pivots = False
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    LU_data = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    unpack_data = True
    unpack_pivots = True
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    LU_data = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    unpack_data = False
    unpack_pivots = True
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    LU_data = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    unpack_data = True
    unpack_pivots = False
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    LU_data = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0],
                           [7.0, 8.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    unpack_data = False
    unpack_pivots = False
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    LU_data = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    unpack_data = True
    unpack_pivots = True
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    LU_data = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0],
                           [7.0, 8.0],
                           [9.0, 10.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).numpy()
    unpack_data = False
    unpack_pivots = True
    
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": unpack_data,
        "unpack_pivots": unpack_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lu_unpack"] = lu_unpack_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lu_unpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_unpack'.")


check_valid('torch.lu_unpack', generated_inputs['torch.lu_unpack'], lib="torch", suffix=0)

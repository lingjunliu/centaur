
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def lu_unpack_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    LU = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]]).numpy()
    Piv = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    LU = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]]).numpy()
    Piv = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    LU = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]]).numpy()
    Piv = torch.tensor([4, 3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    LU = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]]).numpy()
    Piv = torch.tensor([2, 1, 3], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    LU = torch.ones((4, 4)).numpy()
    Piv = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    LU = torch.tensor([[0.0, 1.0],
                      [2.0, 3.0]]).numpy()
    Piv = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    LU = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]]).numpy()
    Piv = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    LU = torch.zeros((3, 3)).numpy()
    Piv = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    LU = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]]).numpy()
    Piv = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    LU = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]]).numpy()
    Piv = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
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

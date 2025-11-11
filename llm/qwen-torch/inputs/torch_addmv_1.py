
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def addmv_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    mat = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]).numpy()
    vec = torch.tensor([10.0, 20.0, 30.0]).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([0.5, 1.5]).numpy()
    mat = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]]).numpy()
    vec = torch.tensor([10.0, 20.0]).numpy()
    beta = 0.0
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0]).numpy()
    mat = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]).numpy()
    vec = torch.tensor([10.0, 20.0, 30.0]).numpy()
    beta = 2.0
    alpha = 0.5
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    mat = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]]).numpy()
    vec = torch.tensor([10.0, 20.0, 30.0]).numpy()
    beta = 0.0
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0]).numpy()
    mat = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]]).numpy()
    vec = torch.tensor([10.0, 20.0]).numpy()
    beta = 1.0
    alpha = 0.5
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    mat = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]).numpy()
    vec = torch.tensor([10.0, 20.0, 30.0]).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    mat = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]).numpy()
    vec = torch.tensor([10.0, 20.0, 30.0]).numpy()
    beta = 1.0
    alpha = 0.5
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    mat = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]]).numpy()
    vec = torch.tensor([10.0, 20.0]).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0]).numpy()
    mat = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]]).numpy()
    vec = torch.tensor([10.0, 20.0]).numpy()
    beta = 0.5
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    mat = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]]).numpy()
    vec = torch.tensor([10.0, 20.0, 30.0]).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.addmv_1"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addmv_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv_1'.")


check_valid('torch.addmv', generated_inputs['torch.addmv_1'], lib="torch", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def svdvals_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    driver = None
    out = torch.empty(2).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    A = torch.randn((4, 5)).numpy()
    driver = None
    out = torch.empty(5).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    A = torch.randn((3, 3)).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    driver = None
    out = torch.empty(2).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    A = torch.randn((2, 2)).numpy()
    driver = None
    out = torch.empty(2).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    A = torch.randn((5, 4)).numpy()
    driver = None
    out = torch.empty(4).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()
    driver = None
    out = torch.empty(4).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    A = torch.randn((6, 3)).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.svdvals_1"] = svdvals_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.svdvals_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.svdvals_1'.")


check_valid('torch.linalg.svdvals', generated_inputs['torch.linalg.svdvals_1'], lib="torch", suffix=1)

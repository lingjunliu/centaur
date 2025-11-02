
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def dequantize_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor = torch.tensor([1, 2, 3], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor = torch.tensor([0, 1, 2, 3], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor = torch.tensor([1, 2, 3], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor = torch.tensor([0, 1, 2], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor = torch.tensor([1, 2, 3], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor = torch.tensor([[1, 2, 3]], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor = torch.tensor([0, 1, 2], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor = torch.tensor([1, 2, 3], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.quint8).numpy()
    input_dict = {
        "tensors": [tensor]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dequantize_2"] = dequantize_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dequantize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dequantize_2'.")


check_valid('torch.dequantize', generated_inputs['torch.dequantize_2'], lib="torch", suffix=2)

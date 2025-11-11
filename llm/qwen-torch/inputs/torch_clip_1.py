
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def clip_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((3, 4)).numpy()
    tensor1 = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                            [0.9, 1.0, 1.1, 1.2]]).numpy()
    tensor2 = torch.tensor([[10.0, 20.0, 30.0, 40.0],
                            [50.0, 60.0, 70.0, 80.0],
                            [90.0, 100.0, 110.0, 120.0]]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    tensor1 = torch.tensor([0.1]).numpy()
    tensor2 = torch.tensor([10.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    tensor1 = torch.tensor([0.5]).numpy()
    tensor2 = torch.tensor([10.0, 20.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 5)).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((2, 2)).numpy()
    tensor1 = torch.tensor([0.5, 0.7]).numpy()
    tensor2 = torch.tensor([0.8, 0.9]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.clip_1"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clip_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip_1'.")


check_valid('torch.clip', generated_inputs['torch.clip_1'], lib="torch", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def margin_ranking_loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([2.0, 3.0, 4.0]).numpy()
    target = torch.tensor([1.0, 1.0, 1.0]).numpy()
    margin = 0.5
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input1 = torch.ones((2, 3)).numpy()
    input2 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    target = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    margin = 0.5
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input1 = torch.tensor([[-1.0, -2.0, -3.0]).numpy()
    input2 = torch.tensor([-2.0, -3.0, -4.0]).numpy()
    target = torch.tensor([1.0, 1.0, 1.0]).numpy()
    margin = 0.1
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input1 = torch.tensor([0.0]).numpy()
    input2 = torch.tensor([1.0]).numpy()
    target = torch.tensor([1.0]).numpy()
    margin = -0.5
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input1 = torch.ones((3, 2, 1)).numpy()
    input2 = torch.tensor([[[1.0],
                     [2.0]],
                    [[3.0],
                     [4.0]],
                    [[5.0],
                     [6.0]]]).numpy()
    target = torch.tensor([[[1.0],
                     [1.0]],
                    [[1.0],
                     [1.0]],
                    [[1.0],
                     [1.0]]]).numpy()
    margin = 0.0
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input2 = torch.tensor([2.0, 3.0, 4.0, 5.0]).numpy()
    target = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    margin = 0.2
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input1 = torch.tensor([[-1.0]).numpy()
    input2 = torch.tensor([-2.0]).numpy()
    target = torch.tensor([1.0]).numpy()
    margin = -1.0
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input1 = torch.ones((4, 5)).numpy()
    input2 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                         [6.0, 7.0, 8.0, 9.0, 10.0],
                         [11.0, 12.0, 13.0, 14.0, 15.0],
                         [16.0, 17.0, 18.0, 19.0, 20.0]]).numpy()
    target = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0]]).numpy()
    margin = 0.5
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input1 = torch.tensor([1.0]).numpy()
    input2 = torch.tensor([2.0]).numpy()
    target = torch.tensor([1.0]).numpy()
    margin = 0.1
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input1 = torch.tensor([0.0, 1.0, 2.0]).numpy()
    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.0, 1.0, 1.0]).numpy()
    margin = 0.0
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, valid
    input1 = torch.tensor([[-1.0, -2.0]).numpy()
    input2 = torch.tensor([-2.0, -3.0]).numpy()
    target = torch.tensor([1.0, 1.0]).numpy()
    margin = -0.5
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12, valid
    input1 = torch.ones((3, 4)).numpy()
    input2 = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0, 12.0]]).numpy()
    target = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]).numpy()
    margin = 0.2
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.margin_ranking_loss"] = margin_ranking_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.margin_ranking_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.margin_ranking_loss'.")


check_valid('torch.nn.functional.margin_ranking_loss', generated_inputs['torch.nn.functional.margin_ranking_loss'], lib="torch", suffix=0)

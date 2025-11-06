
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    input2 = torch.tensor([0.0, -1.0, 2.5], dtype=torch.float32).numpy()
    target = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.0, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[1.2, -0.3, 4.0], [2.2, 5.0, -6.3]], dtype=torch.float64).numpy()
    input2 = torch.tensor([[0.1, -1.3, 3.9], [2.0, 4.5, -5.0]], dtype=torch.float64).numpy()
    target = torch.tensor([[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]], dtype=torch.float64).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 1.0, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[[0.5, -0.5, 1.0]], [[-2.0, 3.0, 0.0]]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[[0.2, -0.2, 0.5]], [[-1.0, 2.5, 1.0]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[1.0, 1.0, -1.0]], [[-1.0, 1.0, 1.0]]], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": -0.5, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor(2.5, dtype=torch.float32).numpy()
    input2 = torch.tensor(3.0, dtype=torch.float32).numpy()
    target = torch.tensor(-1.0, dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.2, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]], dtype=torch.float32).reshape(-1).numpy()
    input2 = torch.tensor([[-0.5, -0.5, 1.5, 2.5, 2.5, -3.5, 4.5, -5.5, 6.5, -7.5]], dtype=torch.float32).reshape(-1).numpy()
    target = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.5, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[0.0, -1.0, 2.0, -3.0],
                           [4.5, -5.5, 6.5, -7.5],
                           [8.0, 0.5, -0.5, 1.5]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[1.0, 0.0, 1.5, -2.5],
                           [4.0, -5.0, 7.0, -8.0],
                           [7.5, 1.0, -1.0, 2.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[1.0, -1.0, 1.0, -1.0],
                           [1.0, 1.0, -1.0, -1.0],
                           [-1.0, 1.0, -1.0, 1.0]], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 2.0, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[[1.0, -2.0, 3.0],
                            [4.0, -5.0, 6.0],
                            [7.0, -8.0, 9.0]]], dtype=torch.float64).numpy()
    input2 = torch.tensor([[[0.5, -1.5, 2.5],
                            [3.5, -4.5, 5.5],
                            [6.5, -7.5, 8.5]]], dtype=torch.float64).numpy()
    target = torch.tensor([[[1.0, -1.0, 1.0],
                            [-1.0, 1.0, -1.0],
                            [1.0, 1.0, -1.0]]], dtype=torch.float64).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.0, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[1.0], [2.0], [-3.0], [4.0], [0.0]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[0.5], [2.5], [-2.0], [3.0], [1.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0]], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": -1.0, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[1.0e10, -1.0e10],
                           [3.0e8, -2.0e8]], dtype=torch.float64).numpy()
    input2 = torch.tensor([[0.9e10, -1.1e10],
                           [2.5e8, -2.5e8]], dtype=torch.float64).numpy()
    target = torch.tensor([[1.0, -1.0],
                           [1.0, -1.0]], dtype=torch.float64).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.0, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.zeros(4, dtype=torch.float32).numpy()
    input2 = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32).numpy()
    target = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.75, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[-0.1, 0.2, -0.3]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[0.4, -0.5, 0.6]], dtype=torch.float32).numpy()
    target = torch.tensor([[1.0, -1.0, 1.0]], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 10.0, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]],
                           [[5.0, -6.0], [-7.0, 8.0]]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[[-1.5, 1.5], [2.5, -3.5]],
                           [[4.5, -5.5], [-6.5, 7.5]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[1.0, -1.0], [1.0, -1.0]],
                           [[-1.0, 1.0], [-1.0, 1.0]]], dtype=torch.float32).numpy()
    input_dict = {"input1": input1, "input2": input2, "target": target, "margin": 0.3, "reduction": "mean"}
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

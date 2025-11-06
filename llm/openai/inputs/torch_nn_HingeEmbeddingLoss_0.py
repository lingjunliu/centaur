
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def hinge_embedding_loss_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([0.2, -0.5, 1.2, 0.0, 2.3], dtype=torch.float32).numpy()
    target = torch.tensor([1, -1, 1, -1, 1], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 1.0,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.tensor([[1.5, -0.2, 0.8],
                          [2.2, 1.0, -1.1]], dtype=torch.float64).numpy()
    target = torch.tensor([[1, 1, -1],
                           [-1, 1, -1]], dtype=torch.float64).numpy()
    input_dict = {
        "margin": 0.5,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.tensor([[[0.3, 1.2],
                           [-0.7, 0.0]],
                          [[2.0, -0.4],
                           [0.5, 1.5]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[1, -1],
                            [1, -1]],
                           [[-1, 1],
                            [1, -1]]], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 2.0,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.tensor([[[[0.4, -0.3, 0.9]],
                           [[-1.2, 0.2, 0.0]]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[[1, -1, 1]],
                            [[-1, 1, -1]]]], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 1.5,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (scalar)
    input = torch.tensor(0.3, dtype=torch.float64).numpy()
    target = torch.tensor(1.0, dtype=torch.float64).numpy()
    input_dict = {
        "margin": 1.0,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.tensor([1.0, 1.0, -1.0, 0.5], dtype=torch.float32).numpy()
    target = torch.tensor([-1, -1, 1, 1], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 1.0,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (float16)
    input = torch.tensor([[0.1, -0.4, 0.7, 1.2],
                          [-0.8, 0.0, 0.3, -0.2],
                          [1.5, -1.1, 0.6, -0.5]], dtype=torch.float16).numpy()
    target = torch.tensor([[1.0, -1.0, 1.0, -1.0],
                           [-1.0, 1.0, -1.0, 1.0],
                           [1.0, -1.0, 1.0, -1.0]], dtype=torch.float16).numpy()
    input_dict = {
        "margin": 0.2,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.tensor([[[0.2, -0.1, 0.0],
                           [0.9, 1.1, -0.7],
                           [-0.3, 0.4, 0.5]]], dtype=torch.float64).numpy()
    target = torch.tensor([[[1, -1, -1],
                            [1, 1, -1],
                            [-1, 1, 1]]], dtype=torch.float64).numpy()
    input_dict = {
        "margin": 0.1,
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (5D)
    input = torch.tensor([[[[[2.0, -0.5]]]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[[[1.0, -1.0]]]]], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 3.0,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.tensor([[0.0],
                          [1.8]], dtype=torch.float32).numpy()
    target = torch.tensor([[-1.0],
                           [1.0]], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 2.5,
        "size_average": False,
        "reduce": True,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.tensor([2.0, -2.0, 0.7, -0.9, 3.3, -1.5], dtype=torch.float64).numpy()
    target = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64).numpy()
    input_dict = {
        "margin": 0.75,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.tensor([[[[0.6, -0.2],
                            [1.4, 0.0]]],
                          [[[ -0.3, 0.8],
                             [ -1.1, 2.2]]]], dtype=torch.float32).numpy()
    target = torch.tensor([[[[1.0, -1.0],
                             [1.0, -1.0]]],
                           [[[ -1.0, 1.0],
                              [ -1.0, 1.0]]]], dtype=torch.float32).numpy()
    input_dict = {
        "margin": 1.2,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.HingeEmbeddingLoss"] = hinge_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.HingeEmbeddingLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.HingeEmbeddingLoss'.")


check_valid('torch.nn.HingeEmbeddingLoss', generated_inputs['torch.nn.HingeEmbeddingLoss'], lib="torch", suffix=0)

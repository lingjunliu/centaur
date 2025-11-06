
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cosine_embedding_loss_inputs():
    list_of_inputs = []

    # Input 1: 2D, float32, mixed targets
    input1 = torch.tensor([[1.0, -2.0, 3.0, -4.0],
                           [0.5, 0.5, 0.5, 0.5],
                           [2.0, -1.0, 0.0, 1.5]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[1.5, -1.5, 2.5, -3.5],
                           [0.2, 0.3, 0.4, 0.5],
                           [1.0, -0.5, 0.5, 1.0]], dtype=torch.float32).numpy()
    target = torch.tensor([1, -1, 1], dtype=torch.int64).numpy()
    margin = 0.0
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 2: 2D, float64, sum reduction
    input1 = torch.tensor([[0.1, -0.2, 0.3],
                           [0.9, -0.8, 0.7]], dtype=torch.float64).numpy()
    input2 = torch.tensor([[-0.3, 0.4, -0.5],
                           [0.6, -0.4, 0.2]], dtype=torch.float64).numpy()
    target = torch.tensor([-1.0, 1.0], dtype=torch.float32).numpy()
    margin = 0.5
    size_average = False
    reduce = True
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 3: Single pair as 2D (N=1, D=5), negative margin
    input1 = torch.tensor([[0.3, -1.2, 0.7, 2.0, -0.4]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[-0.2, 1.0, -0.6, -1.5, 0.9]], dtype=torch.float32).numpy()
    target = torch.tensor([1], dtype=torch.int64).numpy()
    margin = -0.3
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 4: 2D with D=1, no reduction
    input1 = torch.tensor([[1.0],
                           [2.0],
                           [-1.5],
                           [0.7]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[0.8],
                           [1.9],
                           [1.2],
                           [-0.5]], dtype=torch.float32).numpy()
    target = torch.tensor([1, 1, -1, -1], dtype=torch.int64).numpy()
    margin = 0.2
    size_average = True
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 5: High dimension 2D (N=1, D=128), mean
    input1 = torch.randn(1, 128, dtype=torch.float32).numpy()
    input2 = torch.randn(1, 128, dtype=torch.float32).numpy()
    target = torch.tensor([-1], dtype=torch.float64).numpy()
    margin = 0.9
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 6: float32, sum reduction
    input1 = torch.tensor([[0.2, -0.1],
                           [1.5, 0.3],
                           [-0.7, 0.9],
                           [0.4, -0.2],
                           [2.0, 1.0]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[0.1, 0.2],
                           [1.4, 0.1],
                           [-0.6, 1.1],
                           [0.6, -0.3],
                           [1.8, 0.9]], dtype=torch.float32).numpy()
    target = torch.tensor([1, -1, 1, -1, 1], dtype=torch.int64).numpy()
    margin = 0.0
    size_average = False
    reduce = True
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 7: 2D, many samples, mean reduction, high margin
    input1 = torch.tensor([[0.5, -0.4, 0.3],
                           [1.0, 2.0, -1.0],
                           [-0.5, -1.2, 0.8],
                           [0.7, 0.2, -0.3],
                           [1.2, -0.9, 0.4],
                           [-0.6, 0.9, -0.1]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[0.4, -0.3, 0.2],
                           [0.9, 2.1, -0.8],
                           [-0.4, -1.1, 0.7],
                           [0.6, 0.1, -0.4],
                           [1.1, -1.0, 0.5],
                           [-0.7, 0.8, -0.2]], dtype=torch.float32).numpy()
    target = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32).numpy()
    margin = 0.99
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 8: 2D normalized rows, no reduction
    a = torch.tensor([[1.0, 2.0, 2.0],
                      [-1.0, -1.0, 1.0],
                      [2.0, 0.0, -2.0]], dtype=torch.float64)
    b = torch.tensor([[2.0, 0.0, 1.0],
                      [1.0, -2.0, -1.0],
                      [-1.0, 1.0, 0.0]], dtype=torch.float64)
    a = (a / a.norm(dim=1, keepdim=True)).numpy()
    b = (b / b.norm(dim=1, keepdim=True)).numpy()
    target = torch.tensor([-1, -1, -1], dtype=torch.int64).numpy()
    margin = 0.1
    size_average = False
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input1": a,
        "input2": b,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 9: 2D with large values, sum reduction
    input1 = torch.tensor([[100.0, -200.0, 300.0, -400.0],
                           [50.0, 60.0, -70.0, 80.0]], dtype=torch.float32).numpy()
    input2 = torch.tensor([[90.0, -210.0, 310.0, -390.0],
                           [55.0, 65.0, -75.0, 85.0]], dtype=torch.float32).numpy()
    target = torch.tensor([1, 1], dtype=torch.int64).numpy()
    margin = 0.0
    size_average = False
    reduce = True
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 10: 2D random, mean reduction, mixed targets
    input1 = torch.randn(7, 5, dtype=torch.float32).numpy()
    input2 = torch.randn(7, 5, dtype=torch.float32).numpy()
    target = torch.tensor([1, -1, 1, -1, 1, -1, 1], dtype=torch.float32).numpy()
    margin = 0.3
    size_average = False
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    # Input 11: 2D double, no reduction, margin=1.0
    input1 = torch.randn(3, 7, dtype=torch.float64).numpy()
    input2 = torch.randn(3, 7, dtype=torch.float64).numpy()
    target = torch.tensor([-1, 1, -1], dtype=torch.int64).numpy()
    margin = 1.0
    size_average = True
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    return list_of_inputs

generated_inputs["torch.nn.functional.cosine_embedding_loss"] = cosine_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.cosine_embedding_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cosine_embedding_loss'.")


check_valid('torch.nn.functional.cosine_embedding_loss', generated_inputs['torch.nn.functional.cosine_embedding_loss'], lib="torch", suffix=0)

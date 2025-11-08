
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def multilabel_margin_loss_inputs():
    list_of_inputs = []

    input = torch.tensor([0.5, -1.2, 3.3, 0.0, 2.2], dtype=torch.float32).numpy()
    target = torch.tensor([1, 3, -1, -1, -1], dtype=torch.int64).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.1, -0.2, 0.3, -0.4],
                          [1.0, 2.0, -1.0, 0.5]], dtype=torch.float64).numpy()
    target = torch.tensor([[0, 2, -1, -1],
                           [3, 1, 2, 0]], dtype=torch.int64).numpy()
    size_average = False
    reduce = True
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.5, -0.5, 1.5, -1.5, 0.2, -0.2],
                          [2.2, -2.2, 0.0, 0.1, -0.1, 1.0],
                          [-0.3, 0.3, -0.7, 0.7, 1.2, -1.2]], dtype=torch.float32).numpy()
    target = torch.tensor([[5, 2, -1, -1, -1, -1],
                           [1, 3, 4, -1, -1, -1],
                           [0, -1, -1, -1, -1, -1]], dtype=torch.int64).numpy()
    size_average = False
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[-3.0]], dtype=torch.float32).squeeze(0).numpy()
    target = torch.tensor([0], dtype=torch.int64).numpy()
    size_average = True
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.9, -0.8, 0.7]], dtype=torch.float32).numpy()
    target = torch.tensor([[2, -1, -1]], dtype=torch.int64).numpy()
    size_average = True
    reduce = True
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.0],
                          [0.2, 0.3, -0.4, 0.5, -0.6],
                          [-1.1, 1.1, -1.2, 1.2, -1.3],
                          [2.0, -2.0, 2.5, -2.5, 3.0]], dtype=torch.float64).numpy()
    target = torch.tensor([[1, 4, -1, -1, -1],
                           [0, 2, 3, -1, -1],
                           [4, 3, 2, 1, 0],
                           [0, -1, -1, -1, -1]], dtype=torch.int64).numpy()
    size_average = False
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([-0.1, -0.2, 0.0, 0.2, 0.4, -0.3, 1.5], dtype=torch.float32).numpy()
    target = torch.tensor([6, 2, 4, -1, -1, -1, -1], dtype=torch.int64).numpy()
    size_average = False
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3],
                          [1.5, -1.4, 1.3, -1.2, 1.1, -1.0, 0.9, -0.8]], dtype=torch.float32).numpy()
    target = torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0],
                           [0, 3, 5, 6, -1, -1, -1, -1]], dtype=torch.int64).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([2.0, -2.0, 0.0, 1.0], dtype=torch.float64).numpy()
    target = torch.tensor([1, 0, -1, -1], dtype=torch.int64).numpy()
    size_average = False
    reduce = True
    reduction = 'sum'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.0, -0.1, 0.2],
                          [1.0, 0.5, -0.5],
                          [-1.0, 2.0, -2.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[0, 1, 2],
                           [2, 0, 1],
                           [1, 2, 0]], dtype=torch.int64).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.3, -0.3],
                          [-0.7, 0.7]], dtype=torch.float32).numpy()
    target = torch.tensor([[1, 0],
                           [0, 1]], dtype=torch.int64).numpy()
    size_average = False
    reduce = False
    reduction = 'none'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    input = torch.tensor([[0.2, -0.4, 0.6, -0.8],
                          [-1.0, 1.2, -1.4, 1.6],
                          [2.0, -2.2, 2.4, -2.6],
                          [0.0, 0.1, -0.1, 0.2],
                          [1.1, -1.1, 1.3, -1.3]], dtype=torch.float32).numpy()
    target = torch.tensor([[3, 0, -1, -1],
                           [1, 2, -1, -1],
                           [0, 1, 2, 3],
                           [2, -1, -1, -1],
                           [1, 3, 0, -1]], dtype=torch.int64).numpy()
    size_average = True
    reduce = True
    reduction = 'mean'
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "target": target,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }))

    return list_of_inputs

generated_inputs["torch.nn.functional.multilabel_margin_loss"] = multilabel_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.multilabel_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multilabel_margin_loss'.")


check_valid('torch.nn.functional.multilabel_margin_loss', generated_inputs['torch.nn.functional.multilabel_margin_loss'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def multi_margin_loss_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([[0.2, -1.0, 0.5],
                          [1.2, 0.0, -0.3]], dtype=torch.float32).numpy()
    target = torch.tensor([2, 0], dtype=torch.long).numpy()
    p = 1
    margin = 1.0
    weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).numpy()
    size_average = True
    reduce = True
    reduction = "mean"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.tensor([[ -1.2,  0.3,  2.0, -0.5,  1.1],
                          [  0.7, -2.3,  0.0,  1.5, -0.1],
                          [  2.2,  0.4, -1.1,  0.6,  0.0]], dtype=torch.float32).numpy()
    target = torch.tensor([1, 3, 0], dtype=torch.long).numpy()
    p = 2
    margin = 0.2
    weight = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8], dtype=torch.float32).numpy()
    size_average = False
    reduce = True
    reduction = "sum"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=torch.float32).numpy()
    target = torch.tensor([3], dtype=torch.long).numpy()
    p = 1
    margin = 2.0
    weight = torch.ones(4, dtype=torch.float32).numpy()
    size_average = True
    reduce = False
    reduction = "none"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.tensor([0.1, -0.2, 1.5, 0.0], dtype=torch.float32).numpy()
    target = torch.tensor(2, dtype=torch.long).numpy()
    p = 2
    margin = 1.0
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32).numpy()
    size_average = True
    reduce = True
    reduction = "mean"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.tensor([[0.5, -0.1, 0.3, -0.2, 1.0, -1.0, 0.7, 0.2, -0.4, 0.9],
                          [-0.3, 0.8, -0.5, 0.6, -0.7, 0.4, -0.2, 1.1, 0.0, -0.9]], dtype=torch.float32).numpy()
    target = torch.tensor([9, 2], dtype=torch.long).numpy()
    p = 1
    margin = 0.5
    weight = torch.full((10,), 0.5, dtype=torch.float32).numpy()
    size_average = False
    reduce = False
    reduction = "mean"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.tensor([[ 0.2, -0.5,  1.0],
                          [-1.0,  0.3,  0.7],
                          [ 0.8,  0.1, -0.2],
                          [ 1.5, -0.4,  0.0],
                          [-0.6,  0.9, -0.1]], dtype=torch.float32).numpy()
    target = torch.tensor([0, 1, 2, 2, 1], dtype=torch.long).numpy()
    p = 1
    margin = 1.5
    weight = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).numpy()
    size_average = True
    reduce = True
    reduction = "sum"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = torch.tensor([[ 1.0, -0.2,  0.3, -1.0],
                          [-0.7,  0.8, -0.1,  0.4],
                          [ 0.2,  1.2, -1.3,  0.0],
                          [ 0.9, -0.8,  0.5,  0.6]], dtype=torch.float32).numpy()
    target = torch.tensor([1, 3, 0, 2], dtype=torch.long).numpy()
    p = 2
    margin = 0.8
    weight = torch.tensor([1.0, 0.7, 2.0, 0.3], dtype=torch.float32).numpy()
    size_average = False
    reduce = False
    reduction = "none"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.tensor([[0.01, -0.01],
                          [-0.02, 0.03],
                          [0.04, -0.05]], dtype=torch.float32).numpy()
    target = torch.tensor([0, 1, 1], dtype=torch.long).numpy()
    p = 1
    margin = 0.01
    weight = torch.tensor([1.0, 3.0], dtype=torch.float32).numpy()
    size_average = True
    reduce = False
    reduction = "mean"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = torch.tensor([[1.5, -0.5, 0.0],
                          [-1.2, 0.8, -0.3]], dtype=torch.float64).numpy()
    target = torch.tensor([2, 1], dtype=torch.long).numpy()
    p = 1
    margin = 1.0
    weight = torch.tensor([0.9, 1.1, 1.3], dtype=torch.float64).numpy()
    size_average = False
    reduce = False
    reduction = "none"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.tensor([-2.0, 0.0, 3.5], dtype=torch.float32).numpy()
    target = torch.tensor(1, dtype=torch.long).numpy()
    p = 1
    margin = 3.0
    weight = torch.tensor([1.0, 2.5, 0.5], dtype=torch.float32).numpy()
    size_average = True
    reduce = True
    reduction = "sum"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.tensor([[0.3, -0.1, 0.2, -0.4, 0.5]], dtype=torch.float64).numpy()
    target = torch.tensor([4], dtype=torch.long).numpy()
    p = 2
    margin = 0.7
    weight = torch.tensor([1.0, 0.5, 1.5, 2.0, 0.8], dtype=torch.float64).numpy()
    size_average = False
    reduce = True
    reduction = "mean"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.tensor([[ 0.1, -0.2,  0.3],
                          [ 0.4,  0.5, -0.6],
                          [-0.7,  0.8,  0.9],
                          [ 1.0, -1.1,  1.2],
                          [-1.3,  1.4, -1.5],
                          [ 1.6, -1.7,  1.8]], dtype=torch.float32).numpy()
    target = torch.tensor([0, 2, 1, 1, 0, 2], dtype=torch.long).numpy()
    p = 1
    margin = 1.0
    weight = torch.ones(3, dtype=torch.float32).numpy()
    size_average = True
    reduce = False
    reduction = "none"
    input_dict = {
        "input": input,
        "target": target,
        "p": p,
        "margin": margin,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.multi_margin_loss"] = multi_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.multi_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.multi_margin_loss'.")


check_valid('torch.nn.functional.multi_margin_loss', generated_inputs['torch.nn.functional.multi_margin_loss'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([[0.1, -1.2, 3.4],
                              [2.1, 0.0, -0.7]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[1, 0, 1],
                               [0, 1, 0]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.tensor([[5.0, -5.0, 0.5, -0.5]], dtype=torch.float64).numpy()
    target_arr = torch.tensor([[1, 0, 1, 0]], dtype=torch.float64).numpy()
    weight_arr = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float64).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.linspace(-2, 2, steps=10, dtype=torch.float32).reshape(5, 2).numpy()
    target_arr = torch.tensor([[1, 0],
                               [0, 1],
                               [1, 1],
                               [0, 0],
                               [1, 0]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([0.7, 1.3], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.tensor([[50.0, -100.0, 25.0],
                              [-60.0, 80.0, -40.0],
                              [0.0, 0.0, 0.0],
                              [100.0, -50.0, -25.0]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[1, 0, 1],
                               [0, 1, 0],
                               [0, 0, 1],
                               [1, 1, 0]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([1.0, 0.1, 2.0], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.zeros((3, 2), dtype=torch.float32).numpy()
    target_arr = torch.zeros((3, 2), dtype=torch.float32).numpy()
    weight_arr = torch.ones(2, dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.tensor([[-1.0],
                              [0.0],
                              [1.0]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[1],
                               [0],
                               [1]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([3.0], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = (torch.randn(4, 5, dtype=torch.float16) * 3).numpy()
    target_arr = torch.randint(low=0, high=2, size=(4, 5), dtype=torch.int64).to(torch.float16).numpy()
    weight_arr = torch.ones(5, dtype=torch.float16).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = -torch.abs(torch.tensor([[0.2, 1.5],
                                         [3.0, 0.7]], dtype=torch.float32)).numpy()
    target_arr = torch.tensor([[1, 1],
                               [0, 1]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([0.5, 0.5], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([[0.001, -0.002, 0.003],
                              [-0.5, 0.25, -0.125]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[0, 1, 0],
                               [1, 0, 1]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([1.2, 0.8, 1.5], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.linspace(-3, 3, steps=70, dtype=torch.float32).reshape(10, 7).numpy()
    target_arr = (torch.arange(70) % 2).to(torch.float32).reshape(10, 7).numpy()
    weight_arr = torch.linspace(0.5, 1.5, steps=7, dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.tensor([[2.0, -1.0, 0.0, 1.5],
                              [-2.5, 2.5, -3.5, 3.5],
                              [0.3, -0.3, 0.6, -0.6]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [1, 1, 0, 0]], dtype=torch.float32).numpy()
    weight_arr = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32).numpy()
    input_dict = {
        "weight": weight_arr,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MultiLabelSoftMarginLoss"] = multilabel_soft_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MultiLabelSoftMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiLabelSoftMarginLoss'.")


check_valid('torch.nn.MultiLabelSoftMarginLoss', generated_inputs['torch.nn.MultiLabelSoftMarginLoss'], lib="torch", suffix=0)

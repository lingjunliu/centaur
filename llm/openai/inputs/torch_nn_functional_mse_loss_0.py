
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def mse_loss_inputs():
    list_of_inputs = []

    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    target = torch.tensor([1.5, 1.5, 2.5], dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": True, "reduce": True, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[-1.5, 0.5, 2.0], [1.0, -1.0, 2.5]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": False, "reduce": True, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.zeros((2, 1, 4), dtype=torch.float32).numpy()
    target = torch.tensor([[[0.0, 1.0, -1.0, 2.0]], [[-0.5, 0.5, 1.5, -1.5]]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": False, "reduce": False, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(3.0, dtype=torch.float32).numpy()
    target = torch.tensor(1.0, dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": True, "reduce": True, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((1, 2, 3, 4), dtype=torch.float32).numpy()
    target = torch.randn((1, 2, 3, 4), dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": False, "reduce": False, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([1e6, -1e6, 1e12, -1e12, 0.0], dtype=torch.float64).numpy()
    target = torch.tensor([0.0, 0.0, 1e12 + 1.0, -1e12 - 1.0, 0.0], dtype=torch.float64).numpy()
    input_dict = {"input": input, "target": target, "size_average": True, "reduce": True, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.float16).numpy()
    target = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float16).numpy()
    input_dict = {"input": input, "target": target, "size_average": True, "reduce": True, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.empty((0,), dtype=torch.float32).numpy()
    target = torch.empty((0,), dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": True, "reduce": True, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=torch.float32).numpy()
    target = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": False, "reduce": False, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((4, 3, 2), dtype=torch.float32).numpy()
    target = torch.randn((4, 3, 2), dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": True, "reduce": True, "reduction": "mean"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[2.0, -1.0, 0.0], [3.0, -3.0, 1.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[1.0, -2.0, 1.0], [2.5, -2.5, 0.5]], dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": False, "reduce": False, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn((2, 3, 4, 5), dtype=torch.float32).numpy()
    target = torch.randn((2, 3, 4, 5), dtype=torch.float32).numpy()
    input_dict = {"input": input, "target": target, "size_average": False, "reduce": True, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.mse_loss"] = mse_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.mse_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.mse_loss'.")


check_valid('torch.nn.functional.mse_loss', generated_inputs['torch.nn.functional.mse_loss'], lib="torch", suffix=0)

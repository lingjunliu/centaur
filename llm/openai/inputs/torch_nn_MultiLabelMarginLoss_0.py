
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multilabelmarginloss_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([[0.1, 0.2, 0.4, 0.8]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[3, 0, -1, 1]], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(True),
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.tensor([
        [0.5, -0.1, 0.0, 1.0, 0.2],
        [-1.2, 0.3, 0.8, -0.7, 0.0]
    ], dtype=torch.float32).numpy()
    target_arr = torch.tensor([
        [3, 0, 1, -1, 2],
        [2, -1, -1, -1, -1]
    ], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(False),
        "reduce": np.bool_(True),
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([0.2, -0.5, 0.9], dtype=torch.float64).numpy()
    target_arr = torch.tensor([2, 0, -1], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(True),
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.tensor([
        [1.0, -1.0],
        [0.0, 0.0],
        [-0.3, 0.7]
    ], dtype=torch.float32).numpy()
    target_arr = torch.tensor([
        [0, -1],
        [1, -1],
        [1, 0]
    ], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(False),
        "reduce": np.bool_(True),
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.tensor([[-0.2, -1.5, -0.7]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[1, -1, -1]], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(True),
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.tensor([
        [0.5, 0.1, -0.3, 1.2],
        [-0.4, 0.9, 0.0, -0.2],
        [1.0, -1.0, 0.5, 0.3],
        [0.2, 0.8, 0.6, -0.6]
    ], dtype=torch.float32).numpy()
    target_arr = torch.tensor([
        [0, 1, -1, -1],
        [3, -1, -1, -1],
        [2, 0, 1, 3],
        [1, 2, -1, 0]
    ], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(True),
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.tensor([1.5, 0.0, -0.2, 0.7, -0.5], dtype=torch.float32).numpy()
    target_arr = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(True),
        "reduction": "mean",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.tensor([[0.1], [-0.1]], dtype=torch.float32).numpy()
    target_arr = torch.tensor([[0], [0]], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(False),
        "reduce": np.bool_(True),
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([
        [0.5, 0.3, -0.1, 0.9, -0.2, 0.0],
        [-0.8, 0.2, 0.4, -0.3, 1.1, -0.5],
        [0.0, -0.1, 0.2, 0.3, -0.4, 0.7]
    ], dtype=torch.float32).numpy()
    target_arr = torch.tensor([
        [3, 0, -1, 2, -1, -1],
        [4, 2, -1, -1, 1, -1],
        [5, -1, -1, -1, -1, -1]
    ], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(False),
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.tensor([
        [10.0, -10.0, 0.0],
        [-1.0, -2.0, -3.0]
    ], dtype=torch.float32).numpy()
    target_arr = torch.tensor([
        [0, 1, 2],
        [2, -1, -1]
    ], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(False),
        "reduce": np.bool_(False),
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.tensor([0.0, 1.0], dtype=torch.float32).numpy()
    target_arr = torch.tensor([1, -1], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(True),
        "reduction": "sum",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.tensor([
        [-0.2, 0.1, 0.5],
        [0.3, 0.3, 0.3],
        [1.0, -0.5, -0.5],
        [-0.9, 0.9, 0.0]
    ], dtype=torch.float64).numpy()
    target_arr = torch.tensor([
        [2, -1, -1],
        [0, 1, 2],
        [0, -1, -1],
        [1, 0, -1]
    ], dtype=torch.long).numpy()
    input_dict = {
        "size_average": np.bool_(True),
        "reduce": np.bool_(False),
        "reduction": "none",
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MultiLabelMarginLoss"] = multilabelmarginloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MultiLabelMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiLabelMarginLoss'.")


check_valid('torch.nn.MultiLabelMarginLoss', generated_inputs['torch.nn.MultiLabelMarginLoss'], lib="torch", suffix=0)

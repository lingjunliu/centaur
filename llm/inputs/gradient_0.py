
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gradient_inputs():
    list_of_inputs = []

    # Case 1: Basic case with no spacing, dim, or edge_order specified
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).numpy()
    input_dict = {"input": input_tensor, "spacing": None, "dim": None, "edge_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2:  Scalar spacing
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32).numpy()
    input_dict = {"input": input_tensor, "spacing": 2.0, "dim": None, "edge_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: List of scalar spacing
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor, "spacing": [2.0, 3.0], "dim": None, "edge_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: List of Tensor spacing
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).numpy()
    spacing_x = torch.tensor([0.0, 1.0]).numpy()
    spacing_y = torch.tensor([0.0, 2.0, 4.0]).numpy()
    input_dict = {"input": input_tensor, "spacing": (spacing_x, spacing_y), "dim": None, "edge_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Specific dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).numpy()
    input_dict = {"input": input_tensor, "spacing": None, "dim": (1,), "edge_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Specific dimension and scalar spacing
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).numpy()
    input_dict = {"input": input_tensor, "spacing": 2.0, "dim": (0,), "edge_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: edge_order = 2
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).numpy()
    input_dict = {"input": input_tensor, "spacing": None, "dim": None, "edge_order": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: dim is int instead of tuple, remove this one, as it clashes
    # input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).numpy()
    # input_dict = {"input": input_tensor, "spacing": None, "dim": 1, "edge_order": 1}
    # list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.gradient"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gradient'.")

check_valid('torch.gradient', generated_inputs['torch.gradient'], lib="torch")

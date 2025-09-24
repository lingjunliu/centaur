
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def CrossEntropyLoss_inputs():
    list_of_inputs = []

    # Example 1: Class indices, unweighted, default reduction
    input_tensor = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randint(0, 5, (3,)).type(torch.LongTensor).numpy()
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "label_smoothing": 0.0,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Class probabilities, unweighted, sum reduction
    input_tensor = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(3, 5).softmax(dim=1).numpy()
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'sum',
        "label_smoothing": 0.0,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Class indices, weighted, 'none' reduction, ignore_index
    input_tensor = torch.randn(2, 4, requires_grad=True).detach().numpy()
    target_tensor = torch.randint(0, 4, (2,)).type(torch.LongTensor).numpy()
    weight_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {
        "weight": weight_tensor,
        "size_average": None,
        "ignore_index": 1,
        "reduce": None,
        "reduction": 'none',
        "label_smoothing": 0.0,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: K-dimensional input, class indices, label smoothing
    input_tensor = torch.randn(2, 3, 4, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randint(0, 3, (2, 4, 5)).type(torch.LongTensor).numpy()
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "label_smoothing": 0.1,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Unbatched input, class indices
    input_tensor = torch.randn(5, requires_grad=True).detach().numpy()
    target_tensor = torch.randint(0, 5, (1,)).type(torch.LongTensor).item()
    target_tensor = np.array(target_tensor)
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "label_smoothing": 0.0,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.CrossEntropyLoss"] = CrossEntropyLoss_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CrossEntropyLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CrossEntropyLoss'.")

check_valid('torch.nn.CrossEntropyLoss', generated_inputs['torch.nn.CrossEntropyLoss'], lib="torch")

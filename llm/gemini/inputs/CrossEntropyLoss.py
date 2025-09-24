
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cross_entropy_loss_inputs():
    generated_inputs = []

    # Case 1: Class indices, no weight, default reduction, no label smoothing
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "mean", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Class probabilities, no weight, sum reduction, label smoothing
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randn(3, 5).softmax(dim=1).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "sum", "label_smoothing": 0.1}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Class indices, with weight, none reduction, ignore_index
    input_tensor = torch.randn(2, 4).numpy()
    target_tensor = torch.randint(0, 4, (2,)).numpy()
    weight_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": weight_tensor, "ignore_index": 1, "reduction": "none", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 4: K-dimensional input, class indices, mean reduction
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    target_tensor = torch.randint(0, 3, (2, 4, 5)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "mean", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Unbatched input, class indices
    input_tensor = torch.randn(5).numpy()
    target_tensor = torch.randint(0, 5, ()).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "mean", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Class probabilities, with weight, none reduction, label smoothing
    input_tensor = torch.randn(2, 3).numpy()
    target_tensor = torch.randn(2, 3).softmax(dim=1).numpy()
    weight_tensor = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": weight_tensor, "ignore_index": -100, "reduction": "none", "label_smoothing": 0.2}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

def prepare_for_loss(input_dict):
    new_input_dict = {}
    new_input_dict['input'] = torch.tensor(input_dict['input'])
    new_input_dict['target'] = torch.tensor(input_dict['target'])
    if input_dict['weight'] is not None:
        new_input_dict['weight'] = torch.tensor(input_dict['weight'])
    else:
        new_input_dict['weight'] = None

    new_input_dict['ignore_index'] = input_dict['ignore_index']
    new_input_dict['reduction'] = input_dict['reduction']
    new_input_dict['label_smoothing'] = input_dict['label_smoothing']
    return new_input_dict
    

generated_inputs = cross_entropy_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CrossEntropyLoss', generated_inputs)

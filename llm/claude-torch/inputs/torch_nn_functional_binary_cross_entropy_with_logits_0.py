
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    input_val = np.array([0.5, -0.3, 1.2], dtype=np.float32)
    target = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_val,
        "target": target,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[0.2, -0.5], [1.0, -1.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    weight = np.array([[1.0, 2.0], [1.5, 0.5]], dtype=np.float32)
    input_dict = {
        "input": input_val,
        "target": target,
        "weight": weight,
        "size_average": None,
        "reduce": None,
        "reduction": "sum",
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([0.8, -0.2, 1.5], dtype=np.float32)
    target = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    pos_weight = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {
        "input": input_val,
        "target": target,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "pos_weight": pos_weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.random.randn(2, 3, 4).astype(np.float32)
    target = np.random.randint(0, 2, (2, 3, 4)).astype(np.float32)
    input_dict = {
        "input": input_val,
        "target": target,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": "none",
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([0.3, -0.7, 0.9], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_val,
        "target": target,
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([-2.0, -1.5, -0.5], dtype=np.float32)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_val,
        "target": target,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "pos_weight": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([2.5, -1.2, 0.0

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.binary_cross_entropy_with_logits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy_with_logits'.")


check_valid('torch.nn.functional.binary_cross_entropy_with_logits', generated_inputs['torch.nn.functional.binary_cross_entropy_with_logits'], lib="torch", suffix=0)

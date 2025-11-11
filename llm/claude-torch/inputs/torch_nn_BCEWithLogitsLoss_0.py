
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def bce_with_logits_loss_inputs():
    list_of_inputs = []
    
    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": None,
        "input": np.random.randn(10).astype(np.float32),
        "target": np.random.randint(0, 2, size=10).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": np.ones(64).astype(np.float32),
        "input": np.random.randn(10, 64).astype(np.float32),
        "target": np.random.randint(0, 2, size=(10, 64)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "pos_weight": None,
        "input": np.random.randn(5, 10).astype(np.float32),
        "target": np.random.randint(0, 2, size=(5, 10)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "pos_weight": None,
        "input": np.random.randn(8, 3).astype(np.float32),
        "target": np.random.randint(0, 2, size=(8, 3)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": np.ones(6).astype(np.float32),
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": None,
        "input": np.random.randn(6, 20).astype(np.float32),
        "target": np.random.randint(0, 2, size=(6, 20)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "pos_weight": None,
        "input": np.random.randn(4, 8).astype(np.float32),
        "target": np.random.randint(0, 2, size=(4, 8)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": None,
        "size_average": False,
        "reduce": False,
        "reduction": 'mean',
        "pos_weight": None,
        "input": np.random.randn(3, 5).astype(np.float32),
        "target": np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "pos_weight": np.array([3.0]).astype(np

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BCEWithLogitsLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCEWithLogitsLoss'.")


check_valid('torch.nn.BCEWithLogitsLoss', generated_inputs['torch.nn.BCEWithLogitsLoss'], lib="torch", suffix=0)

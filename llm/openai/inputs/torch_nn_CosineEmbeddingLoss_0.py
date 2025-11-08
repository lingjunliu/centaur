
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cosine_embedding_loss_inputs():
    list_of_inputs = []

    input1 = torch.randn(5, dtype=torch.float32).numpy()
    input2 = torch.randn(5, dtype=torch.float32).numpy()
    target = np.array(1, dtype=np.int64)
    input_dict = {
        "margin": 0.0,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(3, 5, dtype=torch.float64).numpy()
    input2 = torch.randn(3, 5, dtype=torch.float64).numpy()
    target = np.ones(3, dtype=np.int64)
    input_dict = {
        "margin": 0.5,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(4, 3, dtype=torch.float32).numpy()
    input2 = torch.randn(4, 3, dtype=torch.float32).numpy()
    target = np.array([-1, 1, -1, 1], dtype=np.int64)
    input_dict = {
        "margin": 0.2,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 4, dtype=torch.float64).numpy()
    input2 = torch.randn(2, 4, dtype=torch.float64).numpy()
    target = np.array([1, -1], dtype=np.int64)
    input_dict = {
        "margin": 0.3,
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(5, 2, dtype=torch.float32).numpy()
    input2 = torch.randn(5, 2, dtype=torch.float32).numpy()
    target = np.array([1, -1, 1, -1, 1], dtype=np.int64)
    input_dict = {
        "margin": 0.7,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(3, 8, dtype=torch.float64).numpy()
    input2 = torch.randn(3, 8, dtype=torch.float64).numpy()
    target = np.array([-1, -1, 1], dtype=np.int64)
    input_dict = {
        "margin": -0.3,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(6, 6, dtype=torch.float32).numpy()
    input2 = torch.randn(6, 6, dtype=torch.float32).numpy()
    target = -np.ones(6, dtype=np.int64)
    input_dict = {
        "margin": 1.0,
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(1, 10, dtype=torch.float64).numpy()
    input2 = torch.randn(1, 10, dtype=torch.float64).numpy()
    target = -np.ones(1, dtype=np.int64)
    input_dict = {
        "margin": 0.99,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(7, 3, dtype=torch.float32).numpy()
    input2 = torch.randn(7, 3, dtype=torch.float32).numpy()
    target = np.array([1, -1, 1, 1, -1, -1, 1], dtype=np.int64)
    input_dict = {
        "margin": 0.25,
        "size_average": True,
        "reduce": False,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(3, dtype=torch.float32).numpy()
    input2 = torch.randn(3, dtype=torch.float32).numpy()
    target = np.array(-1, dtype=np.int64)
    input_dict = {
        "margin": 0.0,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(2, 256, dtype=torch.float32).numpy()
    input2 = torch.randn(2, 256, dtype=torch.float32).numpy()
    target = np.array([1, 1], dtype=np.int64)
    input_dict = {
        "margin": 0.4,
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randn(1, dtype=torch.float64).numpy()
    input2 = torch.randn(1, dtype=torch.float64).numpy()
    target = np.array(1, dtype=np.int64)
    input_dict = {
        "margin": 0.1,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.CosineEmbeddingLoss"] = cosine_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CosineEmbeddingLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CosineEmbeddingLoss'.")


check_valid('torch.nn.CosineEmbeddingLoss', generated_inputs['torch.nn.CosineEmbeddingLoss'], lib="torch", suffix=0)

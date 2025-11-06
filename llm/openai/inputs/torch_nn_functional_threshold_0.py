
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []

    input_np = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(0.0),
        "value": np.float32(0.0),
        "inplace": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float64(1.5),
        "value": np.float64(-1.0),
        "inplace": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(2, 2, 3, dtype=torch.float16).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(-0.5),
        "value": np.float32(-0.5),
        "inplace": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.ones((1, 3, 4, 4), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(0.5),
        "value": np.float32(2.0),
        "inplace": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.tensor(3.14, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(3.0),
        "value": np.float32(-7.0),
        "inplace": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.empty((0, 5), dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(0.0),
        "value": np.float32(1.0),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.tensor([float('nan'), float('-inf'), float('inf'), -1.0, 1.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(0.0),
        "value": np.float32(0.5),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(10, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(1000.0),
        "value": np.float32(-9.0),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(-10.0),
        "value": np.float32(999.0),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(2, 1, 2, 1, 2, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(0.1),
        "value": np.float32(-0.1),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.linspace(-5, 5, steps=24, dtype=torch.float64).reshape(4, 6).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float64(0.0),
        "value": np.float64(-5.0),
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.tensor([[0.0, -0.0, 1e-8, -1e-8],
                             [1e10, -1e10, 1e-10, -1e-10]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input_np,
        "threshold": np.float32(0.0),
        "value": np.float32(-0.0),
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.threshold'.")


check_valid('torch.nn.functional.threshold', generated_inputs['torch.nn.functional.threshold'], lib="torch", suffix=0)

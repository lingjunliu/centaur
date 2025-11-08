
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_grad_norm__inputs():
    list_of_inputs = []

    parameters = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(1.0),
        "norm_type": np.float64(2.0),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(0.5),
        "norm_type": np.float64(1.0),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(10.0),
        "norm_type": np.float64(np.inf),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array(3.0, dtype=np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(2.5),
        "norm_type": np.float64(2.0),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.empty((0,), dtype=np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(100.0),
        "norm_type": np.float64(0.5),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([np.nan, 1.0, -1.0], dtype=np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(1.0),
        "norm_type": np.float64(2.0),
        "error_if_nonfinite": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([np.inf, -np.inf, 1.0], dtype=np.float64)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(5.0),
        "norm_type": np.float64(np.inf),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([100.0, -200.0, 300.0, -50.0, 25.0], dtype=np.float16)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(1e-3),
        "norm_type": np.float64(2.0),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(3.14),
        "norm_type": np.float64(3.0),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([1e-12, -1e-12, 2e-12, 0.0], dtype=np.float64)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(1e-12),
        "norm_type": np.float64(2.0),
        "error_if_nonfinite": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(-1.0),
        "norm_type": np.float64(2.0),
        "error_if_nonfinite": np.bool_(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    parameters = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {
        "parameters": parameters,
        "max_norm": np.float64(50.0),
        "norm_type": np.float64(0.75),
        "error_if_nonfinite": np.bool_(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.utils.clip_grad_norm__1"] = clip_grad_norm__inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.clip_grad_norm__1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm__1'.")


check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm__1'], lib="torch", suffix=1)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kldivloss_inputs():
    def softmax_np(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def log_softmax_np(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        shifted = x - x_max
        return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    list_of_inputs = []

    x = np.array([0.5, -1.2, 3.4, 0.0, 2.2], dtype=np.float32)
    y = np.array([1.0, 0.3, -0.7, 0.9, 0.0], dtype=np.float32)
    input_arr = log_softmax_np(x, axis=0)
    target_arr = softmax_np(y, axis=0)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(3, 5).astype(np.float32)
    y = np.random.randn(3, 5).astype(np.float32)
    input_arr = log_softmax_np(x, axis=1)
    target_arr = softmax_np(y, axis=1)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "batchmean",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(4, 7).astype(np.float64)
    y = np.random.randn(4, 7).astype(np.float64)
    input_arr = log_softmax_np(x, axis=1)
    target_arr = softmax_np(y, axis=1)
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 3, 5).astype(np.float32)
    y = np.random.randn(2, 3, 5).astype(np.float32)
    input_arr = log_softmax_np(x, axis=-1)
    target_arr = softmax_np(y, axis=-1)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_arr = log_softmax_np(x, axis=-1)
    target_arr = softmax_np(y, axis=-1)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "batchmean",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 4).astype(np.float32)
    y = np.random.randn(2, 4).astype(np.float32)
    input_arr = log_softmax_np(x, axis=-1)
    target_arr = softmax_np(y, axis=-1)
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 4, 6).astype(np.float32)
    y = np.random.randn(2, 4, 6).astype(np.float32)
    input_arr = log_softmax_np(x, axis=-1)
    target_arr = log_softmax_np(y, axis=-1)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "log_target": True,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(5, 3).astype(np.float64)
    y = np.random.randn(5, 3).astype(np.float64)
    input_arr = log_softmax_np(x, axis=-1)
    target_arr = log_softmax_np(y, axis=-1)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "batchmean",
        "log_target": True,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(0.0, dtype=np.float32)
    target_arr = np.array(1.0, dtype=np.float32)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    input_arr = log_softmax_np(x, axis=-1).astype(np.float16)
    target_arr = softmax_np(y, axis=-1).astype(np.float16)
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 1, 2, 2, 3).astype(np.float32)
    y = np.random.randn(2, 1, 2, 2, 3).astype(np.float32)
    input_arr = log_softmax_np(x, axis=-1)
    target_arr = softmax_np(y, axis=-1)
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "none",
        "log_target": False,
        "input": input_arr,
        "target": target_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.KLDivLoss"] = kldivloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.KLDivLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.KLDivLoss'.")


check_valid('torch.nn.KLDivLoss', generated_inputs['torch.nn.KLDivLoss'], lib="torch", suffix=0)

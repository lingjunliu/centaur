
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def flatten_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy()
    start_dim = np.int64(0)
    end_dim = np.int64(-1)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.arange(6, dtype=torch.int64).reshape(2, 3).numpy()
    start_dim = np.int32(0)
    end_dim = np.int32(1)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int8).numpy()
    start_dim = np.int64(1)
    end_dim = np.int64(2)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(3, 4, 5, dtype=torch.float64).numpy()
    start_dim = np.int32(-2)
    end_dim = np.int32(-1)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = (torch.randint(0, 2, (2, 3, 4, 5)) > 0).numpy()
    start_dim = np.int64(0)
    end_dim = np.int64(2)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.arange(24, dtype=torch.int32).reshape(1, 2, 3, 4).numpy()
    start_dim = np.int32(2)
    end_dim = np.int32(2)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.tensor(42).numpy()
    start_dim = np.int64(0)
    end_dim = np.int64(-1)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    start_dim = np.int32(0)
    end_dim = np.int32(-1)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.zeros((2, 0, 3), dtype=torch.int16).numpy()
    start_dim = np.int64(1)
    end_dim = np.int64(2)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(1, 2, 3, 1, 4, dtype=torch.float16).numpy()
    start_dim = np.int32(1)
    end_dim = np.int32(3)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.arange(4*5*6*7, dtype=torch.uint8).reshape(4, 5, 6, 7).numpy()
    start_dim = np.int64(-3)
    end_dim = np.int64(-2)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(10, 3, 32, 32, dtype=torch.float32).numpy()
    start_dim = np.int32(1)
    end_dim = np.int32(-1)
    input_dict = {"input": input_arr, "start_dim": start_dim, "end_dim": end_dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.flatten"] = flatten_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.flatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flatten'.")


check_valid('torch.flatten', generated_inputs['torch.flatten'], lib="torch", suffix=0)

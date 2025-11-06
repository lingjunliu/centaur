
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def one_hot_inputs():
    list_of_inputs = []

    input = torch.tensor([0, 1, 2, 3], dtype=torch.long).numpy()
    num_classes = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long).numpy()
    num_classes = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = torch.tensor(2, dtype=torch.long).numpy()
    num_classes = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = torch.tensor(
        [
            [[0, 1, 2, 0], [2, 2, 1, 1], [1, 0, 0, 2]],
            [[2, 1, 0, 1], [0, 0, 2, 2], [1, 2, 1, 0]],
        ],
        dtype=torch.long,
    ).numpy()
    num_classes = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    num_classes = 10
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.empty((0,), dtype=np.int64)
    num_classes = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.empty((2, 0), dtype=np.int64)
    num_classes = 4
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
    num_classes = 2
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    num_classes = 6
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.array(
        [
            [[[0, 1], [2, 0], [1, 1]]],
            [[[2, 2], [0, 1], [0, 0]]],
        ],
        dtype=np.int64,
    )
    num_classes = 3
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.zeros((3,), dtype=np.int64)
    num_classes = 1000
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    input = np.array([4, 4, 0], dtype=np.int64)
    num_classes = 5
    list_of_inputs.append(copy.deepcopy({"input": input, "num_classes": num_classes}))

    return list_of_inputs

generated_inputs["torch.nn.functional.one_hot"] = one_hot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.one_hot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.one_hot'.")


check_valid('torch.nn.functional.one_hot', generated_inputs['torch.nn.functional.one_hot'], lib="torch", suffix=0)

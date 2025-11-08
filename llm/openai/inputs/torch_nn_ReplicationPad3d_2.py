
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def replicationpad3d_inputs():
    list_of_inputs = []

    padding = (1, 1, 1, 1, 1, 1)
    inp = torch.randn(1, 1, 2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (0, 0, 0, 0, 0, 0)
    inp = torch.randn(3, 5, 4, 6, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (2, 0, 3, 1, 1, 0)
    inp = torch.randn(2, 4, 3, 5, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (5, 5, 0, 0, 2, 2)
    inp = torch.randn(1, 2, 1, 2, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (1, 1, 1, 1, 1, 1)
    inp = torch.ones(4, 1, 1, 1, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (0, 2, 1, 0, 4, 3)
    inp = torch.randint(0, 100, (3, 2, 4, 5, 6), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (0, 0, 0, 0, 2, 2)
    inp = torch.randn(5, 6, 2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (3, 3, 6, 6, 1, 1)
    inp = torch.randn(16, 3, 8, 320, 480, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (4, 1, 0, 2, 0, 0)
    base_np = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    inp = torch.tensor(base_np).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (10, 0, 0, 0, 0, 0)
    inp = torch.randn(1, 1, 1, 1, 1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    padding = (1, 0, 2, 0, 3, 0)
    inp = torch.ones(2, 3, 4, 5, 6, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": inp}))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replicationpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_2'.")


check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_2'], lib="torch", suffix=2)

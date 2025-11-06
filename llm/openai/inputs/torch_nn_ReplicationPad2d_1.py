
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def replicationpad2d_inputs():
    list_of_inputs = []

    inp = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    input_dict = {"padding": 1, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 4, 5, dtype=torch.float64).numpy()
    input_dict = {"padding": 2, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.arange(4, dtype=torch.int64).reshape(1, 2, 2).numpy()
    input_dict = {"padding": 1, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.linspace(0, 1, steps=12, dtype=torch.float32).reshape(3, 1, 4).numpy()
    input_dict = {"padding": 0, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(-128, 128, (4, 1, 5, 2)).to(torch.int8).numpy()
    input_dict = {"padding": 1, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 10, 10, dtype=torch.float16).numpy()
    input_dict = {"padding": 3, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 3, 7, 7, dtype=torch.float32).numpy()
    input_dict = {"padding": 0, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(-1000, 1000, (4, 6, 8)).to(torch.int32).numpy()
    input_dict = {"padding": 5, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(0, 256, (2, 1, 2, 3)).to(torch.uint8).numpy()
    input_dict = {"padding": 1, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(-32768, 32767, (3, 3, 1)).to(torch.int16).numpy()
    input_dict = {"padding": 2, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.tensor([[[[0.0, float('nan'), 2.0],
                           [float('inf'), -float('inf'), 5.0],
                           [6.0, 7.0, 8.0]]]], dtype=torch.float32)
    inp = base.numpy()
    input_dict = {"padding": 2, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.arange(24, dtype=torch.int64).reshape(1, 1, 4, 6).numpy()
    input_dict = {"padding": 2, "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad2d_1"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_1'.")


check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_1'], lib="torch", suffix=1)

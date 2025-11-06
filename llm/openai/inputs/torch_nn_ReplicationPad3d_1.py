
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def replicationpad3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(2, 3, 4, 5, 6).numpy()
    padding = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 2
    input_arr = torch.randn(3, 2, 2, 2, dtype=torch.float64).numpy()
    padding = np.int64(1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 3
    input_arr = torch.randint(-50, 50, (1, 1, 3, 3, 3), dtype=torch.int64).to(torch.int32).numpy()
    padding = np.int8(2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 4
    input_arr = torch.ones(4, 1, 1, 1, 1, dtype=torch.float32).numpy()
    padding = np.uint8(3)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 5
    input_arr = torch.arange(1*4*2*5, dtype=torch.float32).reshape(1, 4, 2, 5).numpy()
    padding = np.int16(4)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 6
    input_arr = torch.randn(8, 2, 2, 2, 2, dtype=torch.float16).numpy()
    padding = np.int64(5)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 7
    input_arr = torch.randint(0, 256, (16, 10, 8, 4), dtype=torch.int64).to(torch.int16).numpy()
    padding = np.int32(6)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 8
    input_arr = torch.randint(0, 256, (1, 3, 1, 8, 2), dtype=torch.int64).to(torch.uint8).numpy()
    padding = np.int64(7)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 9
    input_arr = torch.randn(1, 64, 32, 128, 256, dtype=torch.float32).numpy()
    padding = np.int32(1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 10
    input_arr = torch.randn(1, 1, 1, 2, 3, dtype=torch.float64).numpy()
    padding = np.int64(10)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 11
    input_arr = torch.randn(2, 4, 4, 1, dtype=torch.float32).numpy()
    padding = np.int16(9)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # 12
    input_arr = torch.linspace(0, 1, steps=2*2*2*2*2, dtype=torch.float32).reshape(2, 2, 2, 2, 2).numpy()
    padding = np.int64(2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replicationpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad3d_1'.")


check_valid('torch.nn.ReplicationPad3d', generated_inputs['torch.nn.ReplicationPad3d_1'], lib="torch", suffix=1)

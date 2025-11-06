
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def channel_shuffle_inputs():
    list_of_inputs = []

    inp = torch.linspace(-1.0, 1.0, steps=1*8*3*3, dtype=torch.float64).reshape(1, 8, 3, 3).numpy()
    groups = np.int32(4)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.arange(-300, -300 + 5*6*10, dtype=torch.int64).reshape(5, 6, 10).numpy()
    groups = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.randint(-100, 100, (2, 12, 2, 2, 2), dtype=torch.int64).to(torch.int32).numpy()
    groups = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.randint(0, 2, (3, 2, 4, 4), dtype=torch.int64).bool().numpy()
    groups = np.int16(2)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.randint(-128, 127, (1, 4, 2), dtype=torch.int8).numpy()
    groups = np.int8(2)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = (torch.randn(7, 5, 3, dtype=torch.float16) * 10).numpy()
    groups = np.int32(1)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.randint(0, 256, (2, 7, 1, 1), dtype=torch.uint8).numpy()
    groups = np.int64(7)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.arange(1*16*1*2, dtype=torch.float32).reshape(1, 16, 1, 2).numpy()
    groups = np.int32(8)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.zeros((0, 4, 3, 3), dtype=torch.float32).numpy()
    groups = np.int64(2)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.arange(9, dtype=torch.int16).reshape(1, 9, 1, 1, 1, 1).numpy()
    groups = np.int32(3)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.arange(12, dtype=torch.int64).reshape(2, 6, 1).numpy()
    groups = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    inp = torch.randn(4, 32, 2, 2, dtype=torch.float32).numpy()
    groups = np.int32(4)
    list_of_inputs.append(copy.deepcopy({"input": inp, "groups": groups}))

    return list_of_inputs

generated_inputs["torch.channel_shuffle"] = channel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.channel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.channel_shuffle'.")


check_valid('torch.channel_shuffle', generated_inputs['torch.channel_shuffle'], lib="torch", suffix=0)

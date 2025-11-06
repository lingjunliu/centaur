
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def torch_gt_inputs():
    list_of_inputs = []

    input = torch.tensor([-3, 0, 2, 5], dtype=torch.int32).numpy()
    other = np.float32(0.0)
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy()
    other = 2.5
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[-2, -1], [0, 1]], [[2, 3], [-3, -4]]], dtype=torch.int16).numpy()
    other = -1.0
    out = np.empty_like(input, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor(3.1415, dtype=torch.float64).numpy()
    other = np.float64(3.0)
    out = np.empty((), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.randn(2, 3, 1, 4, dtype=torch.float16).numpy()
    other = np.float16(0.5)
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[float('nan'), float('inf')], [-float('inf'), 0.0]], dtype=torch.float32).numpy()
    other = 0.0
    out = np.empty_like(input, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([0, 255, 128], dtype=torch.uint8).numpy()
    other = np.float64(127.5)
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.empty((0, 5), dtype=torch.float32).numpy()
    other = -10.0
    out = np.empty((0, 5), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.arange(-6, 6, dtype=torch.int64).reshape(1, 2, 1, 2, 3).numpy()
    other = 1.0
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([[[-0.2, 0.0, 0.3, -1.5]], [[2.2, -3.3, 4.4, 5.5]], [[-6.6, 7.7, -8.8, 9.9]]], dtype=torch.float32).numpy()
    other = -0.1
    out = np.empty_like(input, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([42.0], dtype=torch.float64).numpy()
    other = np.float64(41.999)
    out = torch.tensor([], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    input = torch.tensor([-2147483648, 2147483647, 0], dtype=torch.int32).numpy()
    other = 0.0
    out = np.empty_like(input, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))

    return list_of_inputs

generated_inputs["torch.gt_2"] = torch_gt_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gt_2'.")


check_valid('torch.gt', generated_inputs['torch.gt_2'], lib="torch", suffix=2)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def squeeze_inputs():
    list_of_inputs = []

    inp = torch.arange(2*1*3*1, dtype=torch.float32).reshape(2,1,3,1).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(1*3*4, dtype=torch.int64).reshape(1,3,4).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(3*4*1, dtype=torch.float16).reshape(3,4,1).numpy()
    dim = -1
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.tensor([[[[True, False]], [[False, True]]]], dtype=torch.bool).reshape(1,2,2,1).numpy()
    dim = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.tensor([[[1+2j, 3+4j]], [[5+6j, 7+8j]]], dtype=torch.complex64).reshape(2,1,2).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.tensor([42], dtype=torch.int32).numpy()
    dim = 0
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(6, dtype=torch.float64).reshape(2,3).numpy()
    dim = 1
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(1*2*3*1, dtype=torch.uint8).reshape(1,2,3,1).numpy()
    dim = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(5*1*1, dtype=torch.int64).reshape(5,1,1).numpy()
    dim = -2
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.ones((1,1,1), dtype=torch.float64).numpy()
    dim = 2
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(4*1*3*1*2*1, dtype=torch.float32).reshape(4,1,3,1,2,1).numpy()
    dim = -4
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    inp = torch.arange(2*1*2*1*2, dtype=torch.float32).reshape(2,1,2,1,2).numpy()
    dim = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "dim": dim}))

    return list_of_inputs

generated_inputs["torch.squeeze_2"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.squeeze_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.squeeze_2'.")


check_valid('torch.squeeze', generated_inputs['torch.squeeze_2'], lib="torch", suffix=2)

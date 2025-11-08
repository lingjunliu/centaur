
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def pixel_unshuffle_inputs():
    list_of_inputs = []

    inp = torch.randn(1, 3, 8, 8, dtype=torch.float32).numpy()
    r = 2
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.arange(3 * 6 * 6, dtype=torch.int64).reshape(3, 6, 6).numpy()
    r = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.randint(0, 2, (2, 1, 4, 4), dtype=torch.bool).numpy()
    r = 2
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    size = 2 * 4 * 3 * 12 * 6
    inp = torch.linspace(-10.0, 10.0, steps=size, dtype=torch.float64).reshape(2, 4, 3, 12, 6).numpy()
    r = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.randint(-1000, 1000, (4, 5, 16, 12), dtype=torch.int64).to(torch.int32).numpy()
    r = 4
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.randint(0, 256, (1, 4, 4), dtype=torch.uint8).numpy()
    r = 2
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.randint(-128, 128, (3, 7, 9, 3), dtype=torch.int64).to(torch.int8).numpy()
    r = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.randn(2, 8, 10, 14, dtype=torch.float16).numpy()
    r = 2
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.linspace(-50.0, 50.0, steps=1 * 2 * 12 * 8, dtype=torch.float64).reshape(1, 2, 12, 8).numpy()
    r = 4
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.arange(6 * 8 * 4, dtype=torch.int16).reshape(6, 8, 4).numpy()
    r = 4
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = (torch.arange(3 * 1 * 2 * 6 * 6) % 2 == 0).reshape(3, 1, 2, 6, 6).to(torch.bool).numpy()
    r = 3
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    inp = torch.randn(5, 2, 18, 12, dtype=torch.float32).numpy()
    r = 6
    list_of_inputs.append(copy.deepcopy({"input": inp, "downscale_factor": r}))

    return list_of_inputs

generated_inputs["torch.nn.functional.pixel_unshuffle"] = pixel_unshuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pixel_unshuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pixel_unshuffle'.")


check_valid('torch.nn.functional.pixel_unshuffle', generated_inputs['torch.nn.functional.pixel_unshuffle'], lib="torch", suffix=0)

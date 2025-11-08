
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

_orig_put = torch.put
def _patched_put(input, index, source, accumulate=False):
    prev = torch.are_deterministic_algorithms_enabled()
    if prev:
        torch.use_deterministic_algorithms(False)
    try:
        return _orig_put(input, index, source, accumulate)
    finally:
        if prev:
            torch.use_deterministic_algorithms(True)
torch.put = _patched_put

def put_inputs():
    list_of_inputs = []

    input_arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).numpy()
    index = torch.tensor([0, 3], dtype=torch.long).numpy()
    source = torch.tensor([-10.0, -40.0], dtype=torch.float32).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.tensor([[1.5, 2.5, -3.5],
                              [4.5, -5.5, 6.5]], dtype=torch.float64).numpy()
    index = torch.tensor([1, 4, 5], dtype=torch.long).numpy()
    source = torch.tensor([10.5, -2.25, 99.0], dtype=torch.float64).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.zeros((2, 2, 3), dtype=torch.int64).numpy()
    index = torch.tensor([0, 5, 11, 3], dtype=torch.long).numpy()
    source = torch.tensor([-1, -2, -3, -4], dtype=torch.int64).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.empty((0,), dtype=torch.float32).numpy()
    index = torch.empty((0,), dtype=torch.long).numpy()
    source = torch.empty((0,), dtype=torch.float32).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.arange(8, dtype=torch.float16).view(2, 1, 2, 2).numpy()
    index = torch.tensor([0, 2, 4, 6], dtype=torch.long).numpy()
    source = torch.tensor([0.5, -0.5, 1.5, -1.5], dtype=torch.float16).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.zeros((3, 3), dtype=torch.bool).numpy()
    index = torch.tensor([0, 1, 8, 4], dtype=torch.long).numpy()
    source = torch.tensor([True, False, True, True], dtype=torch.bool).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.tensor([10, 20, 30, 40], dtype=torch.uint8).numpy()
    index = torch.tensor([1, 3, 2], dtype=torch.long).numpy()
    source = torch.tensor([255, 10, 128], dtype=torch.uint8).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.tensor([[1+0j, 2+0j],
                              [3+0j, 4+0j]], dtype=torch.complex64).numpy()
    index = torch.tensor([0, 3], dtype=torch.long).numpy()
    source = torch.tensor([1+2j, -3+0.5j], dtype=torch.complex64).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.zeros((5, 5), dtype=torch.int32).numpy()
    index = torch.tensor([0, 6, 12, 18, 24], dtype=torch.long).numpy()
    source = torch.tensor([-100, -50, 0, 50, 100], dtype=torch.int32).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    input_arr = torch.arange(10, dtype=torch.float32).numpy()
    index = torch.tensor([2, 4, 6], dtype=torch.long).numpy()
    source = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    accumulate = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "index": index, "source": source, "accumulate": accumulate}))

    return list_of_inputs

generated_inputs["torch.put"] = put_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.put' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.put'.")


check_valid('torch.put', generated_inputs['torch.put'], lib="torch", suffix=0)

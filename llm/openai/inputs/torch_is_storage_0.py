
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def is_storage_inputs():
    list_of_inputs = []

    obj = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[-1, 0, 2], [5, -6, 7]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor(42, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.empty(0, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.randint(0, 256, (2, 3, 4), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.randn(4, dtype=torch.cfloat).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.randn(2, 2, 2, 2, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.arange(10, dtype=torch.int32).view(2, 5)[:, ::2].numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.arange(12, dtype=torch.float64).view(3, 4).t().numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([-2**31, 2**31 - 1], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    return list_of_inputs

generated_inputs["torch.is_storage"] = is_storage_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_storage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_storage'.")


check_valid('torch.is_storage', generated_inputs['torch.is_storage'], lib="torch", suffix=0)

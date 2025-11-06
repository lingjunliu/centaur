
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def typename_inputs():
    list_of_inputs = []

    obj = torch.tensor([1.0, -2.5, 3.3], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[1, 2, -3], [4, -5, 6]], dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.array(-123, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = torch.empty((0,), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.arange(20, dtype=np.float32)[::2]
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.arange(12, dtype=np.float64).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.array([1 + 2j, -3 + 0j, 0 - 1j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = (np.ones((2, 3, 4, 5), dtype=np.float16) * -7.5)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.random.randint(0, 256, size=(4, 4, 3)).astype(np.uint8)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.empty((0, 3), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    obj = np.arange(-48, -48 + 2 * 3 * 4 * 1).reshape(2, 3, 4, 1).astype(np.int16)
    list_of_inputs.append(copy.deepcopy({"obj": obj}))

    return list_of_inputs

generated_inputs["torch.typename"] = typename_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.typename' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.typename'.")


check_valid('torch.typename', generated_inputs['torch.typename'], lib="torch", suffix=0)

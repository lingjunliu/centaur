
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pack_sequence_inputs():
    list_of_inputs = []

    sequences = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(True)}))

    sequences = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(False)}))

    sequences = (np.arange(4, dtype=np.int64).reshape(4, 1) * -2) - 1
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(True)}))

    sequences = (np.arange(5 * 3, dtype=np.float64).reshape(5, 3) - 7.5)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(False)}))

    sequences = np.arange(3 * 2 * 2, dtype=np.float16).reshape(3, 2, 2)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(True)}))

    sequences = (np.arange(6 * 2, dtype=np.int32).reshape(6, 2) - 5)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(True)}))

    sequences = (np.arange(3 * 7 * 1, dtype=np.int32).reshape(3, 7, 1) % 2).astype(bool)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(False)}))

    sequences = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(True)}))

    sequences = np.arange(4 * 4, dtype=np.uint8).reshape(4, 4)
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(False)}))

    sequences = np.arange(3 * 2 * 2 * 2, dtype=np.float32).reshape(3, 2, 2, 2) - 3.0
    list_of_inputs.append(copy.deepcopy({"sequences": sequences, "enforce_sorted": np.bool_(True)}))

    return list_of_inputs

generated_inputs["torch.nn.utils.rnn.pack_sequence"] = pack_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.rnn.pack_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_sequence'.")


check_valid('torch.nn.utils.rnn.pack_sequence', generated_inputs['torch.nn.utils.rnn.pack_sequence'], lib="torch", suffix=0)

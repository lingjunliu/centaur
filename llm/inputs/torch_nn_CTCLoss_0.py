
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ctcloss_inputs():
    list_of_inputs = []

    # Input 1: Basic example with padded targets
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=1, high=20, size=(16, 30), dtype=np.int64)
    input_lengths = tuple(np.full(shape=(16,), fill_value=50, dtype=np.int64).tolist())
    target_lengths = tuple(np.random.randint(low=10, high=30, size=(16,), dtype=np.int64).tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Unpadded targets
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    target_lengths = np.random.randint(low=1, high=50, size=(16,), dtype=np.int64)
    targets = np.random.randint(low=1, high=20, size=(np.sum(target_lengths),), dtype=np.int64)
    input_lengths = tuple(np.full(shape=(16,), fill_value=50, dtype=np.int64).tolist())
    target_lengths = tuple(target_lengths.tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'sum',
        "zero_infinity": True,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unpadded, unbatched (N=1)
    log_probs = torch.randn(50, 20).log_softmax(1).detach().numpy()
    target_lengths = np.random.randint(low=1, high=50, size=(), dtype=np.int64)
    targets = np.random.randint(low=1, high=20, size=(target_lengths,), dtype=np.int64)
    input_lengths = (50,)
    target_lengths = (target_lengths,)
    input_dict = {
        "blank": 0,
        "reduction": 'none',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different blank label
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=0, high=19, size=(16, 30), dtype=np.int64) + 1 # Ensure no zeros if blank is not 0
    input_lengths = tuple(np.full(shape=(16,), fill_value=50, dtype=np.int64).tolist())
    target_lengths = tuple(np.random.randint(low=10, high=30, size=(16,), dtype=np.int64).tolist())
    input_dict = {
        "blank": 19,
        "reduction": 'mean',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Varying input lengths, padded targets
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=1, high=20, size=(16, 30), dtype=np.int64)
    input_lengths = tuple(np.random.randint(low=30, high=50, size=(16,), dtype=np.int64).tolist())
    target_lengths = tuple(np.random.randint(low=10, high=30, size=(16,), dtype=np.int64).tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: edge case target_length=1
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=1, high=20, size=(16, 1), dtype=np.int64)
    input_lengths = tuple(np.full(shape=(16,), fill_value=50, dtype=np.int64).tolist())
    target_lengths = tuple(np.full(shape=(16,), fill_value=1, dtype=np.int64).tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: zero_infinity=True and some input_lengths are too short
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=1, high=20, size=(16, 30), dtype=np.int64)
    input_lengths = tuple(np.random.randint(low=1, high=30, size=(16,), dtype=np.int64).tolist())
    target_lengths = tuple(np.random.randint(low=25, high=30, size=(16,), dtype=np.int64).tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": True,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: different shape of log_probs (T, C)
    log_probs = torch.randn(50, 20).log_softmax(1).detach().numpy()
    target_lengths = np.array([10], dtype=np.int64)
    targets = np.random.randint(low=1, high=20, size=(10,), dtype=np.int64)
    input_lengths = (50,)
    target_lengths = tuple(target_lengths.tolist())

    input_dict = {
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: No reduction
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=1, high=20, size=(16, 30), dtype=np.int64)
    input_lengths = tuple(np.full(shape=(16,), fill_value=50, dtype=np.int64).tolist())
    target_lengths = tuple(np.random.randint(low=10, high=30, size=(16,), dtype=np.int64).tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'none',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sum reduction
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(low=1, high=20, size=(16, 30), dtype=np.int64)
    input_lengths = tuple(np.full(shape=(16,), fill_value=50, dtype=np.int64).tolist())
    target_lengths = tuple(np.random.randint(low=10, high=30, size=(16,), dtype=np.int64).tolist())
    input_dict = {
        "blank": 0,
        "reduction": 'sum',
        "zero_infinity": False,
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.CTCLoss"] = ctcloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.CTCLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CTCLoss'.")

check_valid('torch.nn.CTCLoss', generated_inputs['torch.nn.CTCLoss'], lib="torch", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def ctc_loss_inputs():
    list_of_inputs = []

    # Input 1
    log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 20, (16, 30), dtype=np.int32)
    input_lengths = np.full((16,), 50, dtype=np.int32)
    target_lengths = np.random.randint(1, 30, (16,), dtype=np.int32)
    blank = 0
    reduction = 'mean'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    log_probs = torch.randn(20, 8, 10).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 10, (8, 15), dtype=np.int32)
    input_lengths = np.full((8,), 20, dtype=np.int32)
    target_lengths = np.random.randint(1, 15, (8,), dtype=np.int32)
    blank = 0
    reduction = 'sum'
    zero_infinity = True
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    log_probs = torch.randn(10, 4, 5).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 5, (4, 5), dtype=np.int32)
    input_lengths = np.full((4,), 10, dtype=np.int32)
    target_lengths = np.full((4,), 5, dtype=np.int32)
    blank = 0
    reduction = 'none'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    log_probs = torch.randn(30, 32, 25).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 25, (32, 20), dtype=np.int32)
    input_lengths = np.full((32,), 30, dtype=np.int32)
    target_lengths = np.random.randint(1, 20, (32,), dtype=np.int32)
    blank = 1
    reduction = 'mean'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    log_probs = torch.randn(40, 64, 30).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 30, (64, 25), dtype=np.int32)
    input_lengths = np.full((64,), 40, dtype=np.int32)
    target_lengths = np.random.randint(1, 25, (64,), dtype=np.int32)
    blank = 2
    reduction = 'sum'
    zero_infinity = True
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    log_probs = torch.randn(60, 128, 35).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 35, (128, 30), dtype=np.int32)
    input_lengths = np.full((128,), 60, dtype=np.int32)
    target_lengths = np.random.randint(1, 30, (128,), dtype=np.int32)
    blank = 3
    reduction = 'none'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 - Example with target_lengths = 1
    log_probs = torch.randn(15, 2, 8).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 8, (2, 1), dtype=np.int32)
    input_lengths = np.full((2,), 15, dtype=np.int32)
    target_lengths = np.ones((2,), dtype=np.int32)
    blank = 0
    reduction = 'mean'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - Different blank value
    log_probs = torch.randn(25, 4, 12).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 12, (4, 3), dtype=np.int32)
    input_lengths = np.full((4,), 25, dtype=np.int32)
    target_lengths = np.random.randint(1, 4, (4,), dtype=np.int32)
    blank = 5
    reduction = 'sum'
    zero_infinity = True
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Edge case: target_length = input_length. Reduce target length to avoid errors
    log_probs = torch.randn(10, 2, 5).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 5, (2, 5), dtype=np.int32)
    input_lengths = np.full((2,), 10, dtype=np.int32)
    target_lengths = np.full((2,), 5, dtype=np.int32)
    blank = 0
    reduction = 'mean'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Edge case: target_length = 0
    log_probs = torch.randn(10, 2, 5).log_softmax(2).detach().numpy()
    targets = np.random.randint(1, 5, (2, 1), dtype=np.int32) #Shape of targets does not matter in this case, it will be reshaped.
    input_lengths = np.full((2,), 10, dtype=np.int32)
    target_lengths = np.zeros((2,), dtype=np.int32)
    blank = 0
    reduction = 'mean'
    zero_infinity = False
    input_dict = {
        "log_probs": log_probs,
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "reduction": reduction,
        "zero_infinity": zero_infinity
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.ctc_loss"] = ctc_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.ctc_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.ctc_loss'.")

check_valid('torch.nn.functional.ctc_loss', generated_inputs['torch.nn.functional.ctc_loss'], lib="torch", suffix=0)

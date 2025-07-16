
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fractional_max_pool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 0
    seed2 = 0
    name = "fractional_max_pool_1"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.rand(2, 20, 20, 3).astype(np.float64)
    pooling_ratio = [1.0, 1.2, 1.5, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = False
    seed = 123
    seed2 = 456
    name = "fractional_max_pool_2"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.randint(0, 100, size=(1, 5, 5, 1)).astype(np.int32)
    pooling_ratio = [1.0, 1.1, 1.3, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = True
    seed = 789
    seed2 = 101
    name = "fractional_max_pool_3"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.randint(-100, 100, size=(4, 8, 8, 2)).astype(np.int64)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 0
    seed2 = 0
    name = "fractional_max_pool_4"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.rand(1, 15, 15, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.0, 1.0, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 1
    seed2 = 2
    name = "fractional_max_pool_5"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.random.rand(3, 7, 9, 4).astype(np.float64)
    pooling_ratio = [1.0, 1.6, 1.8, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = False
    seed = 3
    seed2 = 4
    name = "fractional_max_pool_6"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    value = np.random.rand(1, 25, 25, 1).astype(np.float32)
    pooling_ratio = [1.0, 2.5, 3.0, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = True
    seed = 0
    seed2 = 0
    name = "fractional_max_pool_7"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.randint(-50, 50, size=(2, 12, 12, 3)).astype(np.int32)
    pooling_ratio = [1.0, 1.9, 2.1, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 10
    seed2 = 20
    name = "fractional_max_pool_8"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.randint(-1000, 1000, size=(1, 30, 30, 1)).astype(np.int64)
    pooling_ratio = [1.0, 1.01, 1.02, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = False
    seed = 1
    seed2 = 1
    name = "fractional_max_pool_9"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.rand(5, 4, 6, 2).astype(np.float32)
    pooling_ratio = [1.0, 1.35, 1.65, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = False
    seed = 5
    seed2 = 6
    name = "fractional_max_pool_10"

    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_fractional_max_pool_inputs()
generated_inputs["tf.raw_ops.FractionalMaxPool"] = inputs
def check_valid(api, generated_inputs, lib="tf", suffix=0):
    import tensorflow as tf
    def run_api(api, input_dict, cpu=True, lib="tf"):
        if lib == "torch":
            import torch
            func = eval(api)
        elif lib == "tf":
            func = eval(api)

        if "name" in input_dict:
            name = input_dict["name"]
            del input_dict["name"]
        else:
            name = None

        if 'a_minusb' in input_dict:
            aminusb = input_dict["a_minusb"]
            del input_dict["a_minusb"]
        else:
            aminusb = False

        if 'validate_indices' in input_dict:
            validate_indices = input_dict["validate_indices"]
            del input_dict["validate_indices"]
        else:
            validate_indices = False

        inp = {"args": [], "kwargs": input_dict}

        result = func(**inp["kwargs"])
        return result
    for i, input_dict in enumerate(generated_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FractionalMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalMaxPool'.")

check_valid('tf.raw_ops.FractionalMaxPool', generated_inputs['tf.raw_ops.FractionalMaxPool'], lib="tf", suffix=0)

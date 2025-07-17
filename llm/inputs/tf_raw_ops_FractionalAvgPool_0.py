
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FractionalAvgPool_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float64)
    value = np.random.rand(1, 10, 10, 1).astype(np.float64)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data type (int32)
    value = np.random.randint(0, 100, size=(1, 10, 10, 1), dtype=np.int32)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different data type (int64)
    value = np.random.randint(0, 100, size=(1, 10, 10, 1), dtype=np.int64)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different pooling ratios
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.1, 1.2, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Pseudo random is True
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": True, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Overlapping is True
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": True, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Deterministic is True, with seeds
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": True, "seed": 123, "seed2": 456, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  Larger pooling ratio
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": False, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Deterministic True, seeds 0.
    value = np.random.rand(1, 10, 10, 1).astype(np.float32)
    pooling_ratio = [1.0, 1.4, 1.5, 1.0]
    input_dict = {"value": value, "pooling_ratio": pooling_ratio, "pseudo_random": False, "overlapping": False, "deterministic": True, "seed": 0, "seed2": 0, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FractionalAvgPool"] = tf_raw_ops_FractionalAvgPool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FractionalAvgPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalAvgPool'.")

check_valid('tf.raw_ops.FractionalAvgPool', generated_inputs['tf.raw_ops.FractionalAvgPool'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_fractional_max_pool_inputs():
    list_of_inputs = []

    # 1
    value = np.random.randn(1, 8, 8, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case1",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    value = (np.arange(2*7*5*1).reshape(2, 7, 5, 1) - 10).astype(np.int32)
    pooling_ratio = [1.0, 1.8, 1.3, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case2",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    value = (np.random.randn(1, 9, 4, 2) * 5.0).astype(np.float64)
    pooling_ratio = [1.0, 2.2, 1.7, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": True,
        "seed": 42,
        "seed2": 77,
        "name": "case3",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    value = (np.arange(3*10*10*1).reshape(3, 10, 10, 1) - 50).astype(np.int64)
    pooling_ratio = [1.0, 1.33, 1.67, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": True,
        "deterministic": True,
        "seed": 999,
        "seed2": 1,
        "name": "case4",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    value = np.array([[[[1.0], [-2.0], [3.0]],
                       [[-4.0], [5.0], [-6.0]],
                       [[7.0], [-8.0], [9.0]]]], dtype=np.float32)
    pooling_ratio = [1.0, 1.2, 1.6, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": True,
        "seed": 0,
        "seed2": 0,
        "name": "case5",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    value = (np.random.randn(4, 5, 6, 2)).astype(np.float32)
    pooling_ratio = [1.0, 1.99, 1.01, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": True,
        "seed": 0,
        "seed2": 0,
        "name": "case6",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    value = (np.random.randn(2, 12, 7, 4) * 2.0).astype(np.float64)
    pooling_ratio = [1.0, 3.0, 1.2, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": True,
        "seed": 321,
        "seed2": 654,
        "name": "case7",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    value = np.array([[[[1]], [[2]]],
                      [[[3]], [[4]]]], dtype=np.int32)
    pooling_ratio = [1.0, 1.0, 1.0, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case8",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    value = (np.random.randn(1, 15, 9, 1) * 10 - 5).astype(np.float32)
    pooling_ratio = [1.0, 1.3, 2.7, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": True,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case9",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    value = (np.arange(5*13*13*3).reshape(5, 13, 13, 3) - 1000).astype(np.int64)
    pooling_ratio = [1.0, 2.5, 2.5, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": True,
        "seed": 7,
        "seed2": 11,
        "name": "case10",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    value = (np.random.randn(2, 14, 5, 2) * 0.5).astype(np.float32)
    pooling_ratio = [1.0, 1.01, 4.0, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case11",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    value = (np.random.randn(1, 20, 20, 1) + 1.0).astype(np.float64)
    pooling_ratio = [1.0, 1.9, 1.9, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case12",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FractionalMaxPool"] = tf_fractional_max_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FractionalMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalMaxPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FractionalMaxPool', generated_inputs['tf.raw_ops.FractionalMaxPool'], lib="tf", suffix=0)

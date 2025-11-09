
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_LRN_inputs():
    rs = np.random.RandomState(42)
    list_of_inputs = []

    x1 = np.array([[[[1.0]]]], dtype=np.float32)
    input_dict = {
        "depth_radius": 0,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_case1",
        "input": copy.deepcopy(x1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x2 = rs.randn(2, 2, 2, 3).astype(np.float32)
    input_dict = {
        "depth_radius": 1,
        "bias": 2.0,
        "alpha": 1e-4,
        "beta": 0.75,
        "name": "lrn_case2",
        "input": copy.deepcopy(x2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x3 = (rs.randn(1, 3, 3, 8) * 2 - 1).astype(np.float16)
    input_dict = {
        "depth_radius": 2,
        "bias": 1.0,
        "alpha": 0.5,
        "beta": 1.0,
        "name": "lrn_case3",
        "input": copy.deepcopy(x3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x4 = rs.randn(4, 4, 1, 16).astype(np.float32)
    input_dict = {
        "depth_radius": 5,
        "bias": 1e-4,
        "alpha": 1.5,
        "beta": 0.5,
        "name": "lrn_case4",
        "input": copy.deepcopy(x4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x5 = rs.randn(1, 5, 5, 32).astype(np.float16)
    input_dict = {
        "depth_radius": 0,
        "bias": 10.0,
        "alpha": 1e-3,
        "beta": 0.5,
        "name": "lrn_case5",
        "input": copy.deepcopy(x5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x6 = rs.randn(2, 1, 7, 4).astype(np.float32)
    input_dict = {
        "depth_radius": 3,
        "bias": 0.75,
        "alpha": 2.0,
        "beta": 0.25,
        "name": "lrn_case6",
        "input": copy.deepcopy(x6),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x7 = rs.randn(3, 2, 2, 2).astype(np.float16)
    input_dict = {
        "depth_radius": 1,
        "bias": 1.0,
        "alpha": 0.1,
        "beta": 2.0,
        "name": "lrn_case7",
        "input": copy.deepcopy(x7),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x8 = rs.uniform(-3, 3, size=(1, 2, 3, 5)).astype(np.float32)
    input_dict = {
        "depth_radius": 4,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_case8",
        "input": copy.deepcopy(x8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x9 = rs.randn(2, 3, 1, 10).astype(np.float32)
    input_dict = {
        "depth_radius": 9,
        "bias": 0.9,
        "alpha": 0.01,
        "beta": 0.5,
        "name": "lrn_case9",
        "input": copy.deepcopy(x9),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x10 = rs.randn(1, 1, 10, 3).astype(np.float16)
    input_dict = {
        "depth_radius": 1,
        "bias": 1.0,
        "alpha": 0.75,
        "beta": 0.0,
        "name": "lrn_case10",
        "input": copy.deepcopy(x10),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x11 = (rs.randn(5, 5, 5, 6) * 0.1).astype(np.float32)
    input_dict = {
        "depth_radius": 2,
        "bias": 1e-6,
        "alpha": 1e-3,
        "beta": 0.5,
        "name": "lrn_case11",
        "input": copy.deepcopy(x11),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x12 = rs.randn(2, 2, 2, 64).astype(np.float32)
    input_dict = {
        "depth_radius": 7,
        "bias": 3.0,
        "alpha": 5e-4,
        "beta": 1.5,
        "name": "lrn_case12",
        "input": copy.deepcopy(x12),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LRN"] = tf_raw_ops_LRN_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LRN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LRN'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LRN', generated_inputs['tf.raw_ops.LRN'], lib="tf", suffix=0)

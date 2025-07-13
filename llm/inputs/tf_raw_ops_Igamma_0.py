
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_igamma_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    name = "igamma_basic_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with different values
    a = np.array([0.1, 1.2, 2.3], dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "igamma_float64_diff_vals"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16 with small values
    a = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    x = np.array([0.2, 0.8, 1.2], dtype=np.float16)
    name = "igamma_float16_small"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half with different values
    a = np.array([2.0, 3.0, 4.0], dtype=np.float16)
    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    name = "igamma_half_diff"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 with larger values
    a = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    x = np.array([2.5, 7.5, 12.5], dtype=np.float32)
    name = "igamma_float32_large"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional float32
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    name = "igamma_multidim_float32"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional float64
    a = np.array([[0.1, 1.2], [2.3, 3.4]], dtype=np.float64)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "igamma_multidim_float64"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 with zero values for x
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "igamma_float32_zero_x"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Igamma"] = tf_raw_ops_igamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Igamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Igamma'.")

check_valid('tf.raw_ops.Igamma', generated_inputs['tf.raw_ops.Igamma'], lib="tf", suffix=0)

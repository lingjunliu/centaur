
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1
    shape = [2, 3]
    seed = tf.constant([7, 17], dtype=tf.int32)
    means = tf.constant(0.0, dtype=tf.float32)
    stddevs = tf.constant(1.0, dtype=tf.float32)
    minvals = tf.constant(-2.0, dtype=tf.float32)
    maxvals = tf.constant(2.0, dtype=tf.float32)
    name = "truncated_normal_1"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    shape = [5, 4]
    seed = tf.constant([1, 2], dtype=tf.int32)
    means = tf.constant([0.0], dtype=tf.float32)
    stddevs = tf.constant([1.0], dtype=tf.float32)
    minvals = tf.constant([-2.0], dtype=tf.float32)
    maxvals = tf.constant([2.0], dtype=tf.float32)
    name = "truncated_normal_2"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    shape = [10]
    seed = tf.constant([42, 42], dtype=tf.int32)
    means = tf.constant([0.5], dtype=tf.float32)
    stddevs = tf.constant([0.3], dtype=tf.float32)
    minvals = tf.constant([-1.0], dtype=tf.float32)
    maxvals = tf.constant([1.0], dtype=tf.float32)
    name = "truncated_normal_3"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    shape = [3, 2, 5]
    seed = tf.constant([9, 9], dtype=tf.int32)
    means = tf.constant([[0.0]], dtype=tf.float32)
    stddevs = tf.constant([[1.0]], dtype=tf.float32)
    minvals = tf.constant([[-2.0]], dtype=tf.float32)
    maxvals = tf.constant([[2.0]], dtype=tf.float32)
    name = "truncated_normal_4"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    shape = [2, 2, 2]
    seed = tf.constant([100, 100], dtype=tf.int32)
    means = tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=tf.float32)
    stddevs = tf.constant([[1.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
    minvals = tf.constant([[-2.0, -2.0], [-2.0, -2.0]], dtype=tf.float32)
    maxvals = tf.constant([[2.0, 2.0], [2.0, 2.0]], dtype=tf.float32)
    name = "truncated_normal_5"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    shape = [1]
    seed = tf.constant([5, 5], dtype=tf.int32)
    means = tf.constant([0.5], dtype=tf.float32)
    stddevs = tf.constant([0.3], dtype=tf.float32)
    minvals = tf.constant([-1.0], dtype=tf.float32)
    maxvals = tf.constant([1.0], dtype=tf.float32)
    name = "truncated_normal_6"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    shape = [3, 2]
    seed = tf.constant([10, 20], dtype=tf.int32)
    means = tf.constant([[0.0]], dtype=tf.float32)
    stddevs = tf.constant([[1.0]], dtype=tf.float32)
    minvals = tf.constant([[-2.0]], dtype=tf.float32)
    maxvals = tf.constant([[2.0]], dtype=tf.float32)
    name = "truncated_normal_7"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    shape = [1, 1, 1]
    seed = tf.constant([1, 2], dtype=tf.int32)
    means = tf.constant([[0.0]], dtype=tf.float32)
    stddevs = tf.constant([[1.0]], dtype=tf.float32)
    minvals = tf.constant([[-2.0]], dtype=tf.float32)
    maxvals = tf.constant([[2.0]], dtype=tf.float32)
    name = "truncated_normal_8"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    shape = [10, 5]
    seed = tf.constant([3, 7], dtype=tf.int32)
    means = tf.constant([0.0], dtype=tf.float32)
    stddevs = tf.constant([1.0], dtype=tf.float32)
    minvals = tf.constant([-2.0], dtype=tf.float32)
    maxvals = tf.constant([2.0], dtype=tf.float32)
    name = "truncated_normal_9"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    shape = [2, 3]
    seed = tf.constant([7, 17], dtype=tf.int32)
    means = tf.constant([[0.0, 0.5], [0.5, 0.0]], dtype=tf.float32)
    stddevs = tf.constant([[1.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
    minvals = tf.constant([[-2.0, -2.0], [-2.0, -2.0]], dtype=tf.float32)
    maxvals = tf.constant([[2.0, 2.0], [2.0, 2.0]], dtype=tf.float32)
    name = "truncated_normal_10"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal'], lib="tf", suffix=0)

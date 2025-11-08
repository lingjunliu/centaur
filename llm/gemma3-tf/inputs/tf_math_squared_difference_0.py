
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_squared_difference_inputs():
    list_of_inputs = []
    
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    y = np.array([5, 6, 7, 8], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_1'})

    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_2'})

    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_3'})

    x = np.array([1j, 2j, 3j], dtype=np.complex64)
    y = np.array([4j, 5j, 6j], dtype=np.complex64)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_4'})

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_5'})
    
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4, 5, 6], dtype=np.int64)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_6'})

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_7'})

    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    y = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_8'})
    
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_9'})

    x = np.array([1, 2], dtype=np.int32)
    y = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_10'})
    
    return list_of_inputs

generated_inputs["tf.math.squared_difference"] = tf_math_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.squared_difference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.squared_difference'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.squared_difference', generated_inputs['tf.math.squared_difference'], lib="tf", suffix=0)

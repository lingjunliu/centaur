
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_log10_inputs():
    list_of_inputs = []

    # Input 1: Positive scalar
    x = tf.constant(100.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative scalar (should return NaN)
    x = tf.constant(-10.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero (should return -inf)
    x = tf.constant(0.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive 1D array
    x = tf.constant([1.0, 10.0, 100.0, 1000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive 2D array
    x = tf.constant([[1.0, 10.0], [100.0, 1000.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive 3D array
    x = tf.constant([[[1.0, 10.0], [100.0, 1000.0]], [[0.1, 0.01], [0.001, 0.0001]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with mixed positive and negative values
    x = tf.constant([-1.0, 10.0, -100.0, 1000.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with zeros and positive values
    x = tf.constant([0.0, 1.0, 10.0, 100.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with dtype int32
    x = tf.constant([1, 10, 100, 1000], dtype=tf.int32)
    x = tf.cast(x, tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with dtype float64
    x = tf.constant([1.0, 10.0, 100.0, 1000.0], dtype=tf.float64)
    x = tf.cast(x, tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Tensor with very small positive values
    x = tf.constant([1e-5, 1e-6, 1e-7])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Tensor with some NaN values
    x = tf.constant([1.0, np.nan, 10.0])
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.log10"] = tf_experimental_numpy_log10_inputs()
for i in range(len(generated_inputs["tf.experimental.numpy.log10"])):
    generated_inputs["tf.experimental.numpy.log10"][i]["x"] = generated_inputs["tf.experimental.numpy.log10"][i]["x"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.log10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.log10'.")

check_valid('tf.experimental.numpy.log10', generated_inputs['tf.experimental.numpy.log10'], lib="tf", suffix=0)

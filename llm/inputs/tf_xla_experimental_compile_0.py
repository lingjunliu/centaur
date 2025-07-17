
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_xla_experimental_compile_inputs():
    list_of_inputs = []

    def computation1(x):
        return x + 1.0

    input_dict = {
        "computation": [computation1],
        "inputs": [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation2(x, y):
        return tf.matmul(x, y)

    input_dict = {
        "computation": [computation2],
        "inputs": [np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation3(x, y, z):
        return x + y * z

    input_dict = {
        "computation": [computation3],
        "inputs": [np.array([1], dtype=np.int32).item(), np.array([2], dtype=np.int32).item(), np.array([3], dtype=np.int32).item()]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation4(x):
        return tf.reduce_sum(x)

    input_dict = {
        "computation": [computation4],
        "inputs": [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation5():
        return tf.constant(5.0)

    input_dict = {
        "computation": [computation5],
        "inputs": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation6(x):
      return tf.nn.relu(x)

    input_dict = {
        "computation": [computation6],
        "inputs": [np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation7(x):
        return tf.reshape(x, [2, 2])

    input_dict = {
        "computation": [computation7],
        "inputs": [np.array([1, 2, 3, 4], dtype=np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation8(x, y):
        return tf.concat([x, y], axis=0)

    input_dict = {
        "computation": [computation8],
        "inputs": [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation9(x):
        return tf.transpose(x)

    input_dict = {
        "computation": [computation9],
        "inputs": [np.array([[1, 2], [3, 4]], dtype=np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def computation10(x, y):
        return tf.where(x > 0, x, y)

    input_dict = {
        "computation": [computation10],
        "inputs": [np.array([-1, 2, -3, 4], dtype=np.int32), np.array([5, 6, 7, 8], dtype=np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.xla.experimental.compile"] = tf_xla_experimental_compile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.xla.experimental.compile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.xla.experimental.compile'.")

check_valid('tf.xla.experimental.compile', generated_inputs['tf.xla.experimental.compile'], lib="tf", suffix=0)

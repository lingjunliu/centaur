
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_xla_experimental_compile_inputs():
    list_of_inputs = []

    @tf.function(jit_compile=True)
    def computation1(x):
        return tf.add(x, 1)

    input_dict1 = {
        "computation": [computation1],
        "inputs": [np.array([1, 2, 3], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict1)

    @tf.function(jit_compile=True)
    def computation2(x, y):
        return tf.multiply(x, y)

    input_dict2 = {
        "computation": [computation2],
        "inputs": [np.array([4, 5, 6], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32)]
    }
    list_of_inputs.append(input_dict2)

    @tf.function(jit_compile=True)
    def computation3():
        return tf.constant(10)

    input_dict3 = {
        "computation": [computation3],
        "inputs": []
    }
    list_of_inputs.append(input_dict3)

    @tf.function(jit_compile=True)
    def computation4(x):
        return tf.subtract(x, 5)

    input_dict4 = {
        "computation": [computation4],
        "inputs": [np.array([-1, -2, -3], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict4)

    @tf.function(jit_compile=True)
    def computation5(x, y, z):
        return tf.add(tf.multiply(x, y), z)

    input_dict5 = {
        "computation": [computation5],
        "inputs": [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32), np.array([5, 6], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict5)

    @tf.function(jit_compile=True)
    def computation6(x):
      return tf.reshape(x, [2,2])

    input_dict6 = {
        "computation": [computation6],
        "inputs": [np.array([1,2,3,4], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict6)

    @tf.function(jit_compile=True)
    def computation7(x):
        return tf.math.sin(x)

    input_dict7 = {
        "computation": [computation7],
        "inputs": [np.array([0, np.pi/2, np.pi], dtype=np.float32)]
    }
    list_of_inputs.append(input_dict7)

    @tf.function(jit_compile=True)
    def computation8(x):
        return tf.zeros_like(x)

    input_dict8 = {
        "computation": [computation8],
        "inputs": [np.array([[1, 2], [3, 4]], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict8)

    @tf.function(jit_compile=True)
    def computation9(x, y):
        return tf.concat([x, y], axis=0)

    input_dict9 = {
        "computation": [computation9],
        "inputs": [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict9)

    @tf.function(jit_compile=True)
    def computation10(x):
        return tf.one_hot(x, depth=5)

    input_dict10 = {
        "computation": [computation10],
        "inputs": [np.array([0, 1, 2], dtype=np.int32)]
    }
    list_of_inputs.append(input_dict10)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.xla.experimental.compile"] = tf_xla_experimental_compile_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.xla.experimental.compile' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.xla.experimental.compile'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.xla.experimental.compile', generated_inputs['tf.xla.experimental.compile'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_xla_experimental_compile_inputs():
    """
    Generates a list of valid inputs for the tf.xla.experimental.compile function.
    """
    list_of_inputs = []

    # Input 1: Simple addition with float32 tensors
    comp1 = lambda x, y: x + y
    input_dict_1 = {
        'computation': [comp1],
        'inputs': [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Matrix multiplication
    comp2 = lambda a, b: tf.matmul(a, b)
    input_dict_2 = {
        'computation': [comp2],
        'inputs': [
            np.random.rand(3, 4).astype(np.float32),
            np.random.rand(4, 5).astype(np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single input, trigonometric functions, float64
    comp3 = lambda x: tf.sin(tf.cos(x))
    input_dict_3 = {
        'computation': [comp3],
        'inputs': [np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: No inputs, returns a constant
    comp4 = lambda: tf.constant([1, 2, 3], dtype=tf.int32)
    input_dict_4 = {
        'computation': [comp4],
        'inputs': []
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Multiple outputs with int32 tensors
    comp5 = lambda x, y: [x + y, x - y]
    input_dict_5 = {
        'computation': [comp5],
        'inputs': [
            np.array([10, 20, 30], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Mixed precision (int32 and float32)
    comp6 = lambda i, f: tf.cast(i, tf.float32) * f
    input_dict_6 = {
        'computation': [comp6],
        'inputs': [
            np.arange(6, dtype=np.int32).reshape((2, 3)),
            np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Complex numbers
    comp7 = lambda c: tf.math.conj(c) * c
    input_dict_7 = {
        'computation': [comp7],
        'inputs': [np.array([1+2j, 3-4j, 5+0j], dtype=np.complex64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 3D tensors and division
    comp8 = lambda x, y: x / y
    input_dict_8 = {
        'computation': [comp8],
        'inputs': [
            np.ones((2, 2, 2), dtype=np.float32) * 10,
            np.ones((2, 2, 2), dtype=np.float32) * 2
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Single input with negative values
    comp9 = lambda x: tf.abs(x)
    input_dict_9 = {
        'computation': [comp9],
        'inputs': [
            np.array([[-1], [-5], [-10]], dtype=np.int32)
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Computation with an internal constant
    comp10 = lambda x: x + tf.constant(100.0, dtype=tf.float64)
    input_dict_10 = {
        'computation': [comp10],
        'inputs': [
            np.array([1.5, -2.5, 3.0], dtype=np.float64)
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

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

check_valid('tf.xla.experimental.compile', generated_inputs['tf.xla.experimental.compile'], lib="tf", suffix=0)

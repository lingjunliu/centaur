
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow.compat.v1 as tf
import numpy as np
import copy
from tensorflow.python.ops import gradients_util

# The error "tf.gradients is not supported when eager execution is enabled"
# indicates that the function must be run in a graph context.
# The most direct way to achieve this is to disable eager execution
# for the entire script. This reverts TensorFlow to the graph-based
# behavior of TF1, where tf.gradients was originally designed to work.
tf.disable_eager_execution()

def tf_gradients_inputs():
    """
    Generates a list of valid inputs for the tf.gradients function.
    """
    list_of_inputs = []

    # With eager execution disabled, all tf.constant calls create symbolic
    # tensors in the default graph. tf.gradients will then operate on this
    # graph, resolving the RuntimeError. We also return to using lists for
    # 'tensor_list' arguments, as required by the signature.

    # Input 1: Basic scalar differentiation
    a1 = tf.constant(3.0, dtype=tf.float32)
    b1 = 2.0 * a1
    input_dict_1 = {
        'ys': [b1],
        'xs': [a1],
        'grad_ys': None,
        'name': 'gradients_1',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple xs
    a2 = tf.constant(2.0, dtype=tf.float32)
    b2 = tf.constant(5.0, dtype=tf.float32)
    c2 = 3.0 * a2 + 4.0 * b2
    input_dict_2 = {
        'ys': [c2],
        'xs': [a2, b2],
        'grad_ys': None,
        'name': 'gradients_2',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Non-scalar Tensors
    a3_np = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    a3 = tf.constant(a3_np)
    b3 = tf.reduce_sum(3.0 * a3 * a3)
    input_dict_3 = {
        'ys': [b3],
        'xs': [a3],
        'grad_ys': None,
        'name': 'gradients_3',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Using stop_gradients
    a4 = tf.constant(2.0, dtype=tf.float32)
    b4 = 3.0 * a4
    c4 = a4 + b4
    input_dict_4 = {
        'ys': [c4],
        'xs': [a4, b4],
        'grad_ys': None,
        'name': 'gradients_4',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [b4],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: unconnected_gradients='zero'
    a5 = tf.constant([1., 2.], dtype=tf.float32)
    b5 = tf.constant([3., 4.], dtype=tf.float32)
    c5 = a5 * 2.0
    input_dict_5 = {
        'ys': [c5],
        'xs': [a5, b5],
        'grad_ys': None,
        'name': 'gradients_5',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Multiple ys
    a6 = tf.constant(2.0, dtype=tf.float32)
    b6 = a6 * a6
    c6 = 3.0 * a6
    input_dict_6 = {
        'ys': [b6, c6],
        'xs': [a6],
        'grad_ys': None,
        'name': 'gradients_6',
        'gate_gradients': True,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Different dtype (float64)
    a7 = tf.constant(3.0, dtype=tf.float64)
    b7 = a7 * a7 * a7
    input_dict_7 = {
        'ys': [b7],
        'xs': [a7],
        'grad_ys': None,
        'name': 'gradients_7',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using grad_ys
    a8 = tf.constant([2.0, 3.0], dtype=tf.float32)
    b8 = a8 * a8
    grad_ys8 = [tf.constant([10.0, 1.0], dtype=tf.float32)]
    input_dict_8 = {
        'ys': [b8],
        'xs': [a8],
        'grad_ys': grad_ys8,
        'name': 'gradients_8',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Using aggregation_method
    a9 = tf.constant(2.0, dtype=tf.float32)
    b9 = a9 * 2.0
    c9 = a9 * 3.0
    d9 = b9 + c9
    input_dict_9 = {
        'ys': [d9],
        'xs': [a9],
        'grad_ys': None,
        'name': 'gradients_9',
        'gate_gradients': False,
        'aggregation_method': 'ADD_N',
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 3D Tensor
    a10_np = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    a10 = tf.constant(a10_np)
    b10 = tf.reduce_sum(tf.sin(a10))
    input_dict_10 = {
        'ys': [b10],
        'xs': [a10],
        'grad_ys': None,
        'name': 'gradients_10',
        'gate_gradients': False,
        'aggregation_method': None,
        'stop_gradients': [],
        'unconnected_gradients': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.gradients"] = tf_gradients_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.gradients' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.gradients'.")

check_valid('tf.gradients', generated_inputs['tf.gradients'], lib="tf", suffix=0)

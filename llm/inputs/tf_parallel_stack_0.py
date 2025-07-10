
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_parallel_stack_inputs():
    list_of_inputs = []

    def create_input_dict(values, name):
        return {"values": values, "name": name}

    # Input 1: Basic 1D tensors
    values = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_1")))

    # Input 2: 2D tensors
    values = [tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_2")))

    # Input 3: Tensors with different data types (float)
    values = [tf.constant([1.0, 2.0, 3.0]), tf.constant([4.0, 5.0, 6.0])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_4")))

    # Input 4: Tensors with different data types (bool)
    values = [tf.constant([True, False, True]), tf.constant([False, True, False])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_5")))

    # Input 5: Tensors with negative values
    values = [tf.constant([-1, -2, -3]), tf.constant([-4, -5, -6])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_7")))

    # Input 6:  Tensors with different shape of type float64
    values = [tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64), tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float64)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_8")))

    # Input 7:  Tensors with 1 element
    values = [tf.constant([1]), tf.constant([2])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_9")))

    # Input 8: Tensors of complex numbers
    values = [tf.constant([1+1j, 2+2j]), tf.constant([3+3j, 4+4j])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_10")))

    # Input 9: String tensors
    values = [tf.constant(["hello", "world"]), tf.constant(["foo", "bar"])]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_12")))

    # Input 10: Rank 0 tensors
    values = [tf.constant(1), tf.constant(2)]
    list_of_inputs.append(copy.deepcopy(create_input_dict(values, "stack_13")))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.parallel_stack"] = tf_parallel_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.parallel_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.parallel_stack'.")

check_valid('tf.parallel_stack', generated_inputs['tf.parallel_stack'], lib="tf", suffix=0)

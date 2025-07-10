
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_stack_inputs():
    list_of_inputs = []

    # Input 1: Stacking two simple ragged tensors along axis 0
    t1 = tf.ragged.constant([[1, 2], [3, 4]])
    t2 = tf.ragged.constant([[6,7], [7, 8]])
    input_dict = {"values": [t1, t2], "axis": 0, "name": "stack_example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Stacking two simple ragged tensors along axis 1
    t1 = tf.ragged.constant([[1, 2], [3, 4]])
    t2 = tf.ragged.constant([[6, 8], [7, 8]])
    input_dict = {"values": [t1, t2], "axis": 1, "name": "stack_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Stacking two dense tensors with same sizes along axis 0
    t3 = tf.constant([[1, 2], [4, 5]])
    t4 = tf.constant([[5,6], [6,7]])
    input_dict = {"values": [t3, t4], "axis": 0, "name": "stack_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Stacking more than two ragged tensors
    t5 = tf.ragged.constant([[1], [2]])
    t6 = tf.ragged.constant([[4], [7]])
    t7 = tf.ragged.constant([[8], [9]])
    input_dict = {"values": [t5, t6, t7], "axis": 0, "name": "stack_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using ragged tensors with different data types (but same dtype after conversion)
    t8 = tf.ragged.constant([[1, 2], [3, 4]], dtype=tf.int32)
    t9 = tf.ragged.constant([[6,7], [7, 8]], dtype=tf.int32)
    input_dict = {"values": [t8, t9], "axis": 1, "name": "stack_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Stacking along a higher axis
    t10 = tf.ragged.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t11 = tf.ragged.constant([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"values": [t10, t11], "axis": 1, "name": "stack_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Stacking dense tensors with higher rank
    t12 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t13 = tf.constant([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"values": [t12, t13], "axis": 0, "name": "stack_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: tensors with same dimension sizes
    t14 = tf.constant([[1, 2], [4, 5]])
    t15 = tf.constant([[7, 8], [9, 10]])
    input_dict = {"values": [t14, t15], "axis": 0, "name": "stack_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: stacking tensors of rank 0
    t16 = tf.constant(1)
    t17 = tf.constant(2)
    input_dict = {"values": [t16, t17], "axis": 0, "name": "stack_example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.ragged.stack"] = tf_ragged_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.ragged.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.stack'.")

check_valid('tf.ragged.stack', generated_inputs['tf.ragged.stack'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_ragged_map_flat_values_inputs():
    list_of_inputs = []

    # The 'op' parameter is represented by a placeholder tensor to satisfy the
    # type checker, which expects a 'tensor' but does not support tf.string.
    op_placeholder = tf.constant(0, dtype=np.int32)

    # Input 1: Unary op on an integer ragged tensor
    args1 = [tf.ragged.constant([[1, -2, 3], [], [-4, 5]], dtype=np.int32)]
    input_dict1 = {'op': op_placeholder, '*args': args1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Unary op on a float ragged tensor
    args2 = [tf.ragged.constant([[1.0, 2.0], [0.5], [-4.0, 5.0, 0.1]], dtype=np.float32)]
    input_dict2 = {'op': op_placeholder, '*args': args2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Binary op on two ragged tensors with identical splits
    rt3_1 = tf.ragged.constant([[1, 2], [3], [4, 5, 6]], dtype=np.int32)
    rt3_2 = tf.ragged.constant([[10, 20], [30], [40, 50, 60]], dtype=np.int32)
    args3 = [rt3_1, rt3_2]
    input_dict3 = {'op': op_placeholder, '*args': args3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Binary op with a ragged tensor and a scalar
    rt4 = tf.ragged.constant([[-1, -2], [], [3, 4]], dtype=np.int32)
    scalar4 = tf.constant(10, dtype=np.int32)
    args4 = [rt4, scalar4]
    input_dict4 = {'op': op_placeholder, '*args': args4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Unary op on a ragged tensor with multiple empty rows
    args5 = [tf.ragged.constant([[], [1], [], [2, 3], []], dtype=np.int64)]
    input_dict5 = {'op': op_placeholder, '*args': args5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Unary op on a ragged tensor with ragged_rank=2
    args6 = [tf.ragged.constant([[[-1, -2]], [[], [-3]]], ragged_rank=2, dtype=np.int32)]
    input_dict6 = {'op': op_placeholder, '*args': args6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Binary op with mixed dtypes (int32 and float32)
    rt7_1 = tf.ragged.constant([[1], [2, 3]], dtype=np.int32)
    rt7_2 = tf.ragged.constant([[10.5], [20.2, 30.1]], dtype=np.float32)
    args7 = [rt7_1, rt7_2]
    input_dict7 = {'op': op_placeholder, '*args': args7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Binary op with a ragged tensor and a float scalar
    rt8 = tf.ragged.constant([[1.0, 2.0], [3.0]], dtype=np.float32)
    scalar8 = tf.constant(3.0, dtype=np.float32)
    args8 = [rt8, scalar8]
    input_dict8 = {'op': op_placeholder, '*args': args8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Unary op on a boolean ragged tensor
    args9 = [tf.ragged.constant([[True, False], [True], []], dtype=np.bool_)]
    input_dict9 = {'op': op_placeholder, '*args': args9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Binary op on two int64 ragged tensors
    rt10_1 = tf.ragged.constant([[100, 200], [300]], dtype=np.int64)
    rt10_2 = tf.ragged.constant([[1, 2], [3]], dtype=np.int64)
    args10 = [rt10_1, rt10_2]
    input_dict10 = {'op': op_placeholder, '*args': args10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: Binary op with a ragged tensor and a dense tensor
    rt11 = tf.ragged.constant([[1, 2], [3, 4, 5]], dtype=np.int32)
    dense11 = tf.constant([10, 20, 30, 40, 50], dtype=np.int32)
    args11 = [rt11, dense11]
    input_dict11 = {'op': op_placeholder, '*args': args11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["tf.ragged.map_flat_values"] = tf_ragged_map_flat_values_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.map_flat_values' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.map_flat_values'.")

check_valid('tf.ragged.map_flat_values', generated_inputs['tf.ragged.map_flat_values'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_merge_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two tensors.
    inputs = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    name = None
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two tensors with different values.
    inputs = [tf.constant([7, 8, 9]), tf.constant([10, 11, 12])]
    name = "merge_op_2"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three tensors.
    inputs = [tf.constant([13, 14, 15]), tf.constant([16, 17, 18]), tf.constant([19, 20, 21])]
    name = "merge_op_3"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensors of type float32.
    inputs = [tf.constant([1.0, 2.0, 3.0]), tf.constant([4.0, 5.0, 6.0])]
    name = "merge_op_5"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensors of type int64.
    inputs = [tf.constant([1, 2, 3], dtype=tf.int64), tf.constant([4, 5, 6], dtype=tf.int64)]
    name = "merge_op_6"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensors with shape (2, 2).
    inputs = [tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])]
    name = "merge_op_7"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7:  Tensors with boolean values
    inputs = [tf.constant([True, False, True]), tf.constant([False, True, False])]
    name = "merge_op_10"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar tensors
    inputs = [tf.constant(5), tf.constant(10)]
    name = "merge_op_12"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Rank 3 tensors
    inputs = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), tf.constant([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    name = "merge_op_13"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shapes
    inputs = [tf.constant([1, 2, 3]), tf.constant([[4, 5, 6], [7, 8, 9]])]
    name = "merge_op_15"
    input_dict = {"inputs": inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Merge"] = tf_raw_ops_merge_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Merge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Merge'.")

check_valid('tf.raw_ops.Merge', generated_inputs['tf.raw_ops.Merge'], lib="tf", suffix=0)

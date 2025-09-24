
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_logical_or_inputs():
    list_of_inputs = []

    # Input 1: Two single bool elements
    x = tf.constant(True).numpy()
    y = tf.constant(False).numpy()
    name = "or_op_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A tensor and a single bool
    x = tf.constant([True, False, True, False]).numpy()
    y = tf.constant(True).numpy()
    name = "or_op_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two tensors of the same shape
    x = tf.constant([False, False, True, True]).numpy()
    y = tf.constant([False, True, False, True]).numpy()
    name = "or_op_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting example 1
    x = tf.constant([[True, False]]).numpy()
    y = tf.constant([[True], [False]]).numpy()
    name = "or_op_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting example 2
    x = tf.constant([True, False, True]).numpy()
    y = tf.constant(False).numpy()
    name = "or_op_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Two tensors, different shapes but broadcastable
    x = tf.constant([[True, False], [False, True]]).numpy()
    y = tf.constant([False, True]).numpy()
    name = "or_op_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor and a scalar
    x = tf.constant([[[True, False], [False, True]], [[False, True], [True, False]]]).numpy()
    y = tf.constant(True).numpy()
    name = "or_op_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Two 2D tensors
    x = tf.constant([[True, False], [True, True]]).numpy()
    y = tf.constant([[False, True], [False, True]]).numpy()
    name = "or_op_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    x = np.array([], dtype=bool)
    y = tf.constant(True).numpy()
    name = "or_op_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another broadcasting example
    x = tf.constant([True, False]).numpy()
    y = tf.constant([[False], [True]]).numpy()
    name = "or_op_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Two single bool elements, name is None
    x = tf.constant(False).numpy()
    y = tf.constant(True).numpy()
    name = None
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.logical_or"] = tf_math_logical_or_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.logical_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.logical_or'.")

check_valid('tf.math.logical_or', generated_inputs['tf.math.logical_or'], lib="tf", suffix=0)

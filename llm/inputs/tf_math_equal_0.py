
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic equal tensors
    x = tf.constant([1, 2, 3]).numpy()
    y = tf.constant([1, 2, 3]).numpy()
    name = "basic_equal"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different tensors
    x = tf.constant([1, 2, 3]).numpy()
    y = tf.constant([4, 5, 6]).numpy()
    name = "different"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting with scalar
    x = tf.constant([1, 2, 3]).numpy()
    y = tf.constant(2).numpy()
    name = "broadcast_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with scalar (different type)
    x = tf.constant([1.0, 2.0, 3.0]).numpy()
    y = tf.constant(2.0).numpy()
    name = "broadcast_scalar_float"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Two dimensional tensors
    x = tf.constant([[1, 2], [3, 4]]).numpy()
    y = tf.constant([[1, 2], [3, 5]]).numpy()
    name = "two_dimensional"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Three dimensional tensors
    x = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    y = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 9]]]).numpy()
    name = "three_dimensional"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different data types (int32 and int64)
    x = tf.constant([1, 2, 3], dtype=tf.int32).numpy()
    y = tf.constant([1, 2, 3], dtype=tf.int64).numpy()
    x = x.astype(np.int32)
    y = y.astype(np.int32) #Converting to same type for compatibility
    name = "different_dtypes"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values
    x = tf.constant([-1, -2, -3]).numpy()
    y = tf.constant([-1, -2, -4]).numpy()
    name = "negative_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point numbers
    x = tf.constant([1.1, 2.2, 3.3]).numpy()
    y = tf.constant([1.1, 2.2, 3.4]).numpy()
    name = "floating_point"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensor
    x = tf.constant([]).numpy()
    y = tf.constant([]).numpy()
    name = "empty_tensor"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Bool tensor
    x = tf.constant([True, False, True]).numpy()
    y = tf.constant([True, True, False]).numpy()
    name = "bool_tensor"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Rank 0 Tensor (Scalar)
    x = tf.constant(5).numpy()
    y = tf.constant(5).numpy()
    name = "rank_0_tensor"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.equal"] = tf_math_equal_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.equal'.")

check_valid('tf.math.equal', generated_inputs['tf.math.equal'], lib="tf", suffix=0)

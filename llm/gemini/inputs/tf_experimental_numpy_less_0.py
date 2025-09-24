
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_less_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    x1 = tf.constant(np.array([1, 2, 3]))
    x2 = tf.constant(np.array([2, 2, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Case with negative integers
    x1 = tf.constant(np.array([-1, -2, -3]))
    x2 = tf.constant(np.array([-2, -2, -4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Case with mixed positive and negative integers
    x1 = tf.constant(np.array([-1, 2, -3]))
    x2 = tf.constant(np.array([2, -2, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Case with floating point numbers
    x1 = tf.constant(np.array([1.0, 2.0, 3.0]))
    x2 = tf.constant(np.array([2.0, 2.0, 4.0]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Case with different shapes (broadcasting)
    x1 = tf.constant(np.array([1]))
    x2 = tf.constant(np.array([2, 2, 4]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Case with 2D tensors
    x1 = tf.constant(np.array([[1, 2], [3, 4]]))
    x2 = tf.constant(np.array([[2, 3], [4, 5]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Case with different dtypes (int32 and float32)
    x1 = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    x2 = tf.constant(np.array([2, 2, 4], dtype=np.int32))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Case with all values equal
    x1 = tf.constant(np.array([5, 5, 5]))
    x2 = tf.constant(np.array([5, 5, 5]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Case with boolean values
    x1 = tf.constant(np.array([True, False, True]))
    x2 = tf.constant(np.array([False, True, False]))
    x1 = tf.cast(x1, tf.int32)
    x2 = tf.cast(x2, tf.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10: 3D tensors
    x1 = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    x2 = tf.constant(np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]]))
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = tf.experimental.numpy.array(x1)
    x2 = tf.experimental.numpy.array(x2)

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.experimental.numpy.less"] = tf_experimental_numpy_less_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.less' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.less'.")

check_valid('tf.experimental.numpy.less', generated_inputs['tf.experimental.numpy.less'], lib="tf", suffix=0)

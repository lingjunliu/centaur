
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_roll_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant(np.array([0, 1, 2, 3, 4]))
    shift_tensor = tf.constant(np.array(2))
    axis_tensor = tf.constant(np.array(0))
    name = "roll_example_1"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    shift_tensor = tf.constant(np.array([1, -2]))
    axis_tensor = tf.constant(np.array([0, 1]))
    name = "roll_example_2"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    shift_tensor = tf.constant(np.array([2, -3]))
    axis_tensor = tf.constant(np.array([1, 1]))
    name = "roll_example_3"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    shift_tensor = tf.constant(np.array(1))
    axis_tensor = tf.constant(np.array(0))
    name = "roll_example_4"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    shift_tensor = tf.constant(np.array(-1))
    axis_tensor = tf.constant(np.array(1))
    name = "roll_example_5"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    shift_tensor = tf.constant(np.array(1))
    axis_tensor = tf.constant(np.array(2))
    name = "roll_example_6"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    shift_tensor = tf.constant(np.array([1, -1]))
    axis_tensor = tf.constant(np.array([0, 1]))
    name = "roll_example_7"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant(np.array([1, 2, 3, 4, 5]))
    shift_tensor = tf.constant(np.array(-3))
    axis_tensor = tf.constant(np.array(0))
    name = "roll_example_8"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant(np.array([[1, 2], [3, 4]]))
    shift_tensor = tf.constant(np.array(0))
    axis_tensor = tf.constant(np.array(0))
    name = "roll_example_9"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    shift_tensor = tf.constant(np.array([1, -1, 0]))
    axis_tensor = tf.constant(np.array([0, 1, 2]))
    name = "roll_example_10"
    input_dict = {"input": input_tensor, "shift": shift_tensor, "axis": axis_tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
tf.experimental.numpy.experimental_enable_numpy_behavior()
generated_inputs["tf.roll"] = tf_roll_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.roll' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.roll'.")

check_valid('tf.roll', generated_inputs['tf.roll'], lib="tf", suffix=0)

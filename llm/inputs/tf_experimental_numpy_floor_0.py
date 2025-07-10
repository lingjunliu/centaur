
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_floor_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    x = tf.constant(np.array([1.7, 2.2, 3.5]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float tensor
    x = tf.constant(np.array([-1.7, -2.2, -3.5]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer tensor
    x = tf.constant(np.array([1, 2, 3]), dtype=tf.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float tensor
    x = tf.constant(np.array([[1.7, 2.2], [3.5, 4.1]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float tensor
    x = tf.constant(np.array([[[1.7, 2.2], [3.5, 4.1]], [[-1.2, -2.5], [-3.8, -4.9]]]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float16 tensor
    x = tf.constant(np.array([1.7, 2.2, 3.5]), dtype=tf.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 tensor
    x = tf.constant(np.array([1.7, 2.2, 3.5]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with zeros
    x = tf.constant(np.array([0.0, 0.5, -0.5]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with large values
    x = tf.constant(np.array([1000.7, 2000.2, -3000.5]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with small values close to zero
    x = tf.constant(np.array([0.0001, -0.0001, 0.0005]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}

def convert_to_numpy(input_dict):
    new_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, tf.Tensor):
            new_dict[key] = value.numpy()
        else:
            new_dict[key] = value
    return new_dict

tf.experimental.numpy.experimental_enable_numpy_behavior()

temp_list = []
for input_dict in tf_experimental_numpy_floor_inputs():
  temp_list.append(convert_to_numpy(input_dict))

generated_inputs["tf.experimental.numpy.floor"] = temp_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.floor'.")

check_valid('tf.experimental.numpy.floor', generated_inputs['tf.experimental.numpy.floor'], lib="tf", suffix=0)

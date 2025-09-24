
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_cross_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    name = "cross_product_1"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    name = "cross_product_2"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0], dtype=np.float32)
    name = "cross_product_3"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.array([0, 1, 0], dtype=np.float32)
    b = np.array([0, 0, 1], dtype=np.float32)
    name = "cross_product_4"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([0, 0, 1], dtype=np.float32)
    b = np.array([1, 0, 0], dtype=np.float32)
    name = "cross_product_5"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)
    name = "cross_product_6"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
    name = "cross_product_7"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    a = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    b = np.array([4.5, 5.5, 6.5], dtype=np.float64)
    name = "cross_product_8"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.array([[-1, -2, -3]], dtype=np.float32)
    b = np.array([[4, 5, 6]], dtype=np.float32)
    name = "cross_product_9"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.array([1,2,3], dtype=np.int64)
    b = np.array([4,5,6], dtype=np.int64)
    name = "cross_product_10"
    input_dict = {"a": tf.constant(a).numpy(), "b": tf.constant(b).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.cross"] = tf_linalg_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.cross'.")

check_valid('tf.linalg.cross', generated_inputs['tf.linalg.cross'], lib="tf", suffix=0)

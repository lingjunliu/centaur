
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_categorical_inputs():
    list_of_inputs = []

    # Input 1
    logits = np.array([[0.1, 0.9], [0.6, 0.4]], dtype=np.float32)
    num_samples = np.int32(5)
    seed = np.array([1, 2], dtype=np.int32)
    dtype = tf.int64
    name = "samples1"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    logits = np.array([[0.8, 0.2, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    num_samples = np.int32(10)
    seed = np.array([3, 4], dtype=np.int32)
    dtype = tf.int32
    name = "samples2"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    logits = np.array([[0.9, 0.1]], dtype=np.float32)
    num_samples = np.int32(1)
    seed = np.array([5, 6], dtype=np.int32)
    dtype = tf.int64
    name = "samples3"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    logits = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    num_samples = np.int32(7)
    seed = np.array([7, 8], dtype=np.int32)
    dtype = tf.int32
    name = "samples4"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    logits = np.array([[0.3, 0.7], [0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    num_samples = np.int32(3)
    seed = np.array([9, 10], dtype=np.int32)
    dtype = tf.int64
    name = "samples5"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    logits = np.array([[0.05, 0.15, 0.3, 0.5]], dtype=np.float32)
    num_samples = np.int32(4)
    seed = np.array([11, 12], dtype=np.int32)
    dtype = tf.int32
    name = "samples6"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    logits = np.array([[0.6, 0.4], [0.5, 0.5]], dtype=np.float32)
    num_samples = np.int32(2)
    seed = np.array([13, 14], dtype=np.int64)
    dtype = tf.int64
    name = "samples7"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    logits = np.array([[0.7, 0.3, 0.0, 0.0]], dtype=np.float32)
    num_samples = np.int32(6)
    seed = np.array([15, 16], dtype=np.int64)
    dtype = tf.int32
    name = "samples8"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    logits = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
    num_samples = np.int32(8)
    seed = np.array([17, 18], dtype=np.int64)
    dtype = tf.int64
    name = "samples9"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    logits = np.array([[0.4, 0.6]], dtype=np.float32)
    num_samples = np.int32(9)
    seed = np.array([19, 20], dtype=np.int64)
    dtype = tf.int32
    name = "samples10"
    input_dict = {"logits": logits, "num_samples": num_samples, "seed": seed, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.stateless_categorical"] = tf_random_stateless_categorical_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.stateless_categorical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_categorical'.")

check_valid('tf.random.stateless_categorical', generated_inputs['tf.random.stateless_categorical'], lib="tf", suffix=0)
